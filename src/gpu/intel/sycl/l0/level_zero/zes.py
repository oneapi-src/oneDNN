"""
 Copyright (C) 2019-2021 Intel Corporation

 SPDX-License-Identifier: MIT

 @file zes.py
 @version v1.3-r1.3.7

 """
import platform
from ctypes import *
from enum import *

###############################################################################
__version__ = "1.0"

###############################################################################
## @brief Handle to a driver instance
class zes_driver_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of device object
class zes_device_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle for a Sysman device scheduler queue
class zes_sched_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle for a Sysman device performance factors
class zes_perf_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle for a Sysman device power domain
class zes_pwr_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle for a Sysman device frequency domain
class zes_freq_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle for a Sysman device engine group
class zes_engine_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle for a Sysman device standby control
class zes_standby_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle for a Sysman device firmware
class zes_firmware_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle for a Sysman device memory module
class zes_mem_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle for a Sysman fabric port
class zes_fabric_port_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle for a Sysman device temperature sensor
class zes_temp_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle for a Sysman device power supply
class zes_psu_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle for a Sysman device fan
class zes_fan_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle for a Sysman device LED
class zes_led_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle for a Sysman device RAS error set
class zes_ras_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle for a Sysman device diagnostics test suite
class zes_diag_handle_t(c_void_p):
    pass

###############################################################################
## @brief Defines structure types
class zes_structure_type_v(IntEnum):
    DEVICE_PROPERTIES = 0x1                         ## ::zes_device_properties_t
    PCI_PROPERTIES = 0x2                            ## ::zes_pci_properties_t
    PCI_BAR_PROPERTIES = 0x3                        ## ::zes_pci_bar_properties_t
    DIAG_PROPERTIES = 0x4                           ## ::zes_diag_properties_t
    ENGINE_PROPERTIES = 0x5                         ## ::zes_engine_properties_t
    FABRIC_PORT_PROPERTIES = 0x6                    ## ::zes_fabric_port_properties_t
    FAN_PROPERTIES = 0x7                            ## ::zes_fan_properties_t
    FIRMWARE_PROPERTIES = 0x8                       ## ::zes_firmware_properties_t
    FREQ_PROPERTIES = 0x9                           ## ::zes_freq_properties_t
    LED_PROPERTIES = 0xa                            ## ::zes_led_properties_t
    MEM_PROPERTIES = 0xb                            ## ::zes_mem_properties_t
    PERF_PROPERTIES = 0xc                           ## ::zes_perf_properties_t
    POWER_PROPERTIES = 0xd                          ## ::zes_power_properties_t
    PSU_PROPERTIES = 0xe                            ## ::zes_psu_properties_t
    RAS_PROPERTIES = 0xf                            ## ::zes_ras_properties_t
    SCHED_PROPERTIES = 0x10                         ## ::zes_sched_properties_t
    SCHED_TIMEOUT_PROPERTIES = 0x11                 ## ::zes_sched_timeout_properties_t
    SCHED_TIMESLICE_PROPERTIES = 0x12               ## ::zes_sched_timeslice_properties_t
    STANDBY_PROPERTIES = 0x13                       ## ::zes_standby_properties_t
    TEMP_PROPERTIES = 0x14                          ## ::zes_temp_properties_t
    DEVICE_STATE = 0x15                             ## ::zes_device_state_t
    PROCESS_STATE = 0x16                            ## ::zes_process_state_t
    PCI_STATE = 0x17                                ## ::zes_pci_state_t
    FABRIC_PORT_CONFIG = 0x18                       ## ::zes_fabric_port_config_t
    FABRIC_PORT_STATE = 0x19                        ## ::zes_fabric_port_state_t
    FAN_CONFIG = 0x1a                               ## ::zes_fan_config_t
    FREQ_STATE = 0x1b                               ## ::zes_freq_state_t
    OC_CAPABILITIES = 0x1c                          ## ::zes_oc_capabilities_t
    LED_STATE = 0x1d                                ## ::zes_led_state_t
    MEM_STATE = 0x1e                                ## ::zes_mem_state_t
    PSU_STATE = 0x1f                                ## ::zes_psu_state_t
    BASE_STATE = 0x20                               ## ::zes_base_state_t
    RAS_CONFIG = 0x21                               ## ::zes_ras_config_t
    RAS_STATE = 0x22                                ## ::zes_ras_state_t
    TEMP_CONFIG = 0x23                              ## ::zes_temp_config_t
    PCI_BAR_PROPERTIES_1_2 = 0x24                   ## ::zes_pci_bar_properties_1_2_t

class zes_structure_type_t(c_int):
    def __str__(self):
        return str(zes_structure_type_v(self.value))


###############################################################################
## @brief Base for all properties types
class zes_base_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p)                                             ## [in,out][optional] pointer to extension-specific structure
    ]

###############################################################################
## @brief Base for all descriptor types
class zes_base_desc_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p)                                             ## [in][optional] pointer to extension-specific structure
    ]

###############################################################################
## @brief Base for all state types
class zes_base_state_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p)                                             ## [in][optional] pointer to extension-specific structure
    ]

###############################################################################
## @brief Base for all config types
class zes_base_config_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p)                                             ## [in][optional] pointer to extension-specific structure
    ]

###############################################################################
## @brief Base for all capability types
class zes_base_capability_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p)                                             ## [in][optional] pointer to extension-specific structure
    ]

###############################################################################
## @brief Maximum number of characters in string properties.
ZES_STRING_PROPERTY_SIZE = 64

###############################################################################
## @brief Types of accelerator engines
class zes_engine_type_flags_v(IntEnum):
    OTHER = ZE_BIT(0)                               ## Undefined types of accelerators.
    COMPUTE = ZE_BIT(1)                             ## Engines that process compute kernels only (no 3D content).
    _3D = ZE_BIT(2)                                 ## Engines that process 3D content only (no compute kernels).
    MEDIA = ZE_BIT(3)                               ## Engines that process media workloads.
    DMA = ZE_BIT(4)                                 ## Engines that copy blocks of data.
    RENDER = ZE_BIT(5)                              ## Engines that can process both 3D content and compute kernels.

class zes_engine_type_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Device repair status
class zes_repair_status_v(IntEnum):
    UNSUPPORTED = 0                                 ## The device does not support in-field repairs.
    NOT_PERFORMED = 1                               ## The device has never been repaired.
    PERFORMED = 2                                   ## The device has been repaired.

class zes_repair_status_t(c_int):
    def __str__(self):
        return str(zes_repair_status_v(self.value))


###############################################################################
## @brief Device reset reasons
class zes_reset_reason_flags_v(IntEnum):
    WEDGED = ZE_BIT(0)                              ## The device needs to be reset because one or more parts of the hardware
                                                    ## is wedged
    REPAIR = ZE_BIT(1)                              ## The device needs to be reset in order to complete in-field repairs

class zes_reset_reason_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Device state
class zes_device_state_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("reset", zes_reset_reason_flags_t),                            ## [out] Indicates if the device needs to be reset and for what reasons.
                                                                        ## returns 0 (none) or combination of ::zes_reset_reason_flag_t
        ("repaired", zes_repair_status_t)                               ## [out] Indicates if the device has been repaired
    ]

###############################################################################
## @brief Device properties
class zes_device_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("core", ze_device_properties_t),                               ## [out] Core device properties
        ("numSubdevices", c_ulong),                                     ## [out] Number of sub-devices. A value of 0 indicates that this device
                                                                        ## doesn't have sub-devices.
        ("serialNumber", c_char * ZES_STRING_PROPERTY_SIZE),            ## [out] Manufacturing serial number (NULL terminated string value). Will
                                                                        ## be set to the string "unkown" if this cannot be determined for the
                                                                        ## device.
        ("boardNumber", c_char * ZES_STRING_PROPERTY_SIZE),             ## [out] Manufacturing board number (NULL terminated string value). Will
                                                                        ## be set to the string "unkown" if this cannot be determined for the
                                                                        ## device.
        ("brandName", c_char * ZES_STRING_PROPERTY_SIZE),               ## [out] Brand name of the device (NULL terminated string value). Will be
                                                                        ## set to the string "unkown" if this cannot be determined for the
                                                                        ## device.
        ("modelName", c_char * ZES_STRING_PROPERTY_SIZE),               ## [out] Model name of the device (NULL terminated string value). Will be
                                                                        ## set to the string "unkown" if this cannot be determined for the
                                                                        ## device.
        ("vendorName", c_char * ZES_STRING_PROPERTY_SIZE),              ## [out] Vendor name of the device (NULL terminated string value). Will
                                                                        ## be set to the string "unkown" if this cannot be determined for the
                                                                        ## device.
        ("driverVersion", c_char * ZES_STRING_PROPERTY_SIZE)            ## [out] Installed driver version (NULL terminated string value). Will be
                                                                        ## set to the string "unkown" if this cannot be determined for the
                                                                        ## device.
    ]

###############################################################################
## @brief Contains information about a process that has an open connection with
##        this device
## 
## @details
##     - The application can use the process ID to query the OS for the owner
##       and the path to the executable.
class zes_process_state_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("processId", c_ulong),                                         ## [out] Host OS process ID.
        ("memSize", c_ulonglong),                                       ## [out] Device memory size in bytes allocated by this process (may not
                                                                        ## necessarily be resident on the device at the time of reading).
        ("sharedSize", c_ulonglong),                                    ## [out] The size of shared device memory mapped into this process (may
                                                                        ## not necessarily be resident on the device at the time of reading).
        ("engines", zes_engine_type_flags_t)                            ## [out] Bitfield of accelerator engine types being used by this process.
    ]

###############################################################################
## @brief PCI address
class zes_pci_address_t(Structure):
    _fields_ = [
        ("domain", c_ulong),                                            ## [out] BDF domain
        ("bus", c_ulong),                                               ## [out] BDF bus
        ("device", c_ulong),                                            ## [out] BDF device
        ("function", c_ulong)                                           ## [out] BDF function
    ]

###############################################################################
## @brief PCI speed
class zes_pci_speed_t(Structure):
    _fields_ = [
        ("gen", c_int32_t),                                             ## [out] The link generation. A value of -1 means that this property is
                                                                        ## unknown.
        ("width", c_int32_t),                                           ## [out] The number of lanes. A value of -1 means that this property is
                                                                        ## unknown.
        ("maxBandwidth", c_int64_t)                                     ## [out] The maximum bandwidth in bytes/sec (sum of all lanes). A value
                                                                        ## of -1 means that this property is unknown.
    ]

###############################################################################
## @brief Static PCI properties
class zes_pci_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("address", zes_pci_address_t),                                 ## [out] The BDF address
        ("maxSpeed", zes_pci_speed_t),                                  ## [out] Fastest port configuration supported by the device (sum of all
                                                                        ## lanes)
        ("haveBandwidthCounters", ze_bool_t),                           ## [out] Indicates if ::zes_pci_stats_t.rxCounter and
                                                                        ## ::zes_pci_stats_t.txCounter will have valid values
        ("havePacketCounters", ze_bool_t),                              ## [out] Indicates if ::zes_pci_stats_t.packetCounter will have valid
                                                                        ## values
        ("haveReplayCounters", ze_bool_t)                               ## [out] Indicates if ::zes_pci_stats_t.replayCounter will have valid
                                                                        ## values
    ]

###############################################################################
## @brief PCI link status
class zes_pci_link_status_v(IntEnum):
    UNKNOWN = 0                                     ## The link status could not be determined
    GOOD = 1                                        ## The link is up and operating as expected
    QUALITY_ISSUES = 2                              ## The link is up but has quality and/or bandwidth degradation
    STABILITY_ISSUES = 3                            ## The link has stability issues and preventing workloads making forward
                                                    ## progress

class zes_pci_link_status_t(c_int):
    def __str__(self):
        return str(zes_pci_link_status_v(self.value))


###############################################################################
## @brief PCI link quality degradation reasons
class zes_pci_link_qual_issue_flags_v(IntEnum):
    REPLAYS = ZE_BIT(0)                             ## A significant number of replays are occurring
    SPEED = ZE_BIT(1)                               ## There is a degradation in the maximum bandwidth of the link

class zes_pci_link_qual_issue_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief PCI link stability issues
class zes_pci_link_stab_issue_flags_v(IntEnum):
    RETRAINING = ZE_BIT(0)                          ## Link retraining has occurred to deal with quality issues

class zes_pci_link_stab_issue_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Dynamic PCI state
class zes_pci_state_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("status", zes_pci_link_status_t),                              ## [out] The current status of the port
        ("qualityIssues", zes_pci_link_qual_issue_flags_t),             ## [out] If status is ::ZES_PCI_LINK_STATUS_QUALITY_ISSUES, 
                                                                        ## then this gives a combination of ::zes_pci_link_qual_issue_flag_t for
                                                                        ## quality issues that have been detected;
                                                                        ## otherwise, 0 indicates there are no quality issues with the link at
                                                                        ## this time."
        ("stabilityIssues", zes_pci_link_stab_issue_flags_t),           ## [out] If status is ::ZES_PCI_LINK_STATUS_STABILITY_ISSUES, 
                                                                        ## then this gives a combination of ::zes_pci_link_stab_issue_flag_t for
                                                                        ## reasons for the connection instability;
                                                                        ## otherwise, 0 indicates there are no connection stability issues at
                                                                        ## this time."
        ("speed", zes_pci_speed_t)                                      ## [out] The current port configure speed
    ]

###############################################################################
## @brief PCI bar types
class zes_pci_bar_type_v(IntEnum):
    MMIO = 0                                        ## MMIO registers
    ROM = 1                                         ## ROM aperture
    MEM = 2                                         ## Device memory

class zes_pci_bar_type_t(c_int):
    def __str__(self):
        return str(zes_pci_bar_type_v(self.value))


###############################################################################
## @brief Properties of a pci bar
class zes_pci_bar_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("type", zes_pci_bar_type_t),                                   ## [out] The type of bar
        ("index", c_ulong),                                             ## [out] The index of the bar
        ("base", c_ulonglong),                                          ## [out] Base address of the bar.
        ("size", c_ulonglong)                                           ## [out] Size of the bar.
    ]

###############################################################################
## @brief Properties of a pci bar, including the resizable bar.
class zes_pci_bar_properties_1_2_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("type", zes_pci_bar_type_t),                                   ## [out] The type of bar
        ("index", c_ulong),                                             ## [out] The index of the bar
        ("base", c_ulonglong),                                          ## [out] Base address of the bar.
        ("size", c_ulonglong),                                          ## [out] Size of the bar.
        ("resizableBarSupported", ze_bool_t),                           ## [out] Support for Resizable Bar on this device.
        ("resizableBarEnabled", ze_bool_t)                              ## [out] Resizable Bar enabled on this device
    ]

###############################################################################
## @brief PCI stats counters
## 
## @details
##     - Percent replays is calculated by taking two snapshots (s1, s2) and
##       using the equation: %replay = 10^6 * (s2.replayCounter -
##       s1.replayCounter) / (s2.maxBandwidth * (s2.timestamp - s1.timestamp))
##     - Percent throughput is calculated by taking two snapshots (s1, s2) and
##       using the equation: %bw = 10^6 * ((s2.rxCounter - s1.rxCounter) +
##       (s2.txCounter - s1.txCounter)) / (s2.maxBandwidth * (s2.timestamp -
##       s1.timestamp))
class zes_pci_stats_t(Structure):
    _fields_ = [
        ("timestamp", c_ulonglong),                                     ## [out] Monotonic timestamp counter in microseconds when the measurement
                                                                        ## was made.
                                                                        ## This timestamp should only be used to calculate delta time between
                                                                        ## snapshots of this structure.
                                                                        ## Never take the delta of this timestamp with the timestamp from a
                                                                        ## different structure since they are not guaranteed to have the same base.
                                                                        ## The absolute value of the timestamp is only valid during within the
                                                                        ## application and may be different on the next execution.
        ("replayCounter", c_ulonglong),                                 ## [out] Monotonic counter for the number of replay packets (sum of all
                                                                        ## lanes). Will always be 0 if ::zes_pci_properties_t.haveReplayCounters
                                                                        ## is FALSE.
        ("packetCounter", c_ulonglong),                                 ## [out] Monotonic counter for the number of packets (sum of all lanes).
                                                                        ## Will always be 0 if ::zes_pci_properties_t.havePacketCounters is
                                                                        ## FALSE.
        ("rxCounter", c_ulonglong),                                     ## [out] Monotonic counter for the number of bytes received (sum of all
                                                                        ## lanes). Will always be 0 if
                                                                        ## ::zes_pci_properties_t.haveBandwidthCounters is FALSE.
        ("txCounter", c_ulonglong),                                     ## [out] Monotonic counter for the number of bytes transmitted (including
                                                                        ## replays) (sum of all lanes). Will always be 0 if
                                                                        ## ::zes_pci_properties_t.haveBandwidthCounters is FALSE.
        ("speed", zes_pci_speed_t)                                      ## [out] The current speed of the link (sum of all lanes)
    ]

###############################################################################
## @brief Diagnostic results
class zes_diag_result_v(IntEnum):
    NO_ERRORS = 0                                   ## Diagnostic completed without finding errors to repair
    ABORT = 1                                       ## Diagnostic had problems running tests
    FAIL_CANT_REPAIR = 2                            ## Diagnostic had problems setting up repairs
    REBOOT_FOR_REPAIR = 3                           ## Diagnostics found errors, setup for repair and reboot is required to
                                                    ## complete the process

class zes_diag_result_t(c_int):
    def __str__(self):
        return str(zes_diag_result_v(self.value))


###############################################################################
## @brief Diagnostic test index to use for the very first test.
ZES_DIAG_FIRST_TEST_INDEX = 0x0

###############################################################################
## @brief Diagnostic test index to use for the very last test.
ZES_DIAG_LAST_TEST_INDEX = 0xFFFFFFFF

###############################################################################
## @brief Diagnostic test
class zes_diag_test_t(Structure):
    _fields_ = [
        ("index", c_ulong),                                             ## [out] Index of the test
        ("name", c_char * ZES_STRING_PROPERTY_SIZE)                     ## [out] Name of the test
    ]

###############################################################################
## @brief Diagnostics test suite properties
class zes_diag_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("onSubdevice", ze_bool_t),                                     ## [out] True if the resource is located on a sub-device; false means
                                                                        ## that the resource is on the device of the calling Sysman handle
        ("subdeviceId", c_ulong),                                       ## [out] If onSubdevice is true, this gives the ID of the sub-device
        ("name", c_char * ZES_STRING_PROPERTY_SIZE),                    ## [out] Name of the diagnostics test suite
        ("haveTests", ze_bool_t)                                        ## [out] Indicates if this test suite has individual tests which can be
                                                                        ## run separately (use the function ::zesDiagnosticsGetTests() to get the
                                                                        ## list of these tests)
    ]

###############################################################################
## @brief Accelerator engine groups
class zes_engine_group_v(IntEnum):
    ALL = 0                                         ## Access information about all engines combined.
    COMPUTE_ALL = 1                                 ## Access information about all compute engines combined. Compute engines
                                                    ## can only process compute kernels (no 3D content).
    MEDIA_ALL = 2                                   ## Access information about all media engines combined.
    COPY_ALL = 3                                    ## Access information about all copy (blitter) engines combined.
    COMPUTE_SINGLE = 4                              ## Access information about a single compute engine - this is an engine
                                                    ## that can process compute kernels. Note that single engines may share
                                                    ## the same underlying accelerator resources as other engines so activity
                                                    ## of such an engine may not be indicative of the underlying resource
                                                    ## utilization - use ::ZES_ENGINE_GROUP_3D_RENDER_COMPUTE_ALL for that.
    RENDER_SINGLE = 5                               ## Access information about a single render engine - this is an engine
                                                    ## that can process both 3D content and compute kernels. Note that single
                                                    ## engines may share the same underlying accelerator resources as other
                                                    ## engines so activity of such an engine may not be indicative of the
                                                    ## underlying resource utilization - use
                                                    ## ::ZES_ENGINE_GROUP_3D_RENDER_COMPUTE_ALL for that.
    MEDIA_DECODE_SINGLE = 6                         ## Access information about a single media decode engine. Note that
                                                    ## single engines may share the same underlying accelerator resources as
                                                    ## other engines so activity of such an engine may not be indicative of
                                                    ## the underlying resource utilization - use ::ZES_ENGINE_GROUP_MEDIA_ALL
                                                    ## for that.
    MEDIA_ENCODE_SINGLE = 7                         ## Access information about a single media encode engine. Note that
                                                    ## single engines may share the same underlying accelerator resources as
                                                    ## other engines so activity of such an engine may not be indicative of
                                                    ## the underlying resource utilization - use ::ZES_ENGINE_GROUP_MEDIA_ALL
                                                    ## for that.
    COPY_SINGLE = 8                                 ## Access information about a single media encode engine. Note that
                                                    ## single engines may share the same underlying accelerator resources as
                                                    ## other engines so activity of such an engine may not be indicative of
                                                    ## the underlying resource utilization - use ::ZES_ENGINE_GROUP_COPY_ALL
                                                    ## for that.
    MEDIA_ENHANCEMENT_SINGLE = 9                    ## Access information about a single media enhancement engine. Note that
                                                    ## single engines may share the same underlying accelerator resources as
                                                    ## other engines so activity of such an engine may not be indicative of
                                                    ## the underlying resource utilization - use ::ZES_ENGINE_GROUP_MEDIA_ALL
                                                    ## for that.
    _3D_SINGLE = 10                                 ## Access information about a single 3D engine - this is an engine that
                                                    ## can process 3D content only. Note that single engines may share the
                                                    ## same underlying accelerator resources as other engines so activity of
                                                    ## such an engine may not be indicative of the underlying resource
                                                    ## utilization - use ::ZES_ENGINE_GROUP_3D_RENDER_COMPUTE_ALL for that.
    _3D_RENDER_COMPUTE_ALL = 11                     ## Access information about all 3D/render/compute engines combined.
    RENDER_ALL = 12                                 ## Access information about all render engines combined. Render engines
                                                    ## are those than process both 3D content and compute kernels.
    _3D_ALL = 13                                    ## Access information about all 3D engines combined. 3D engines can
                                                    ## process 3D content only (no compute kernels).

class zes_engine_group_t(c_int):
    def __str__(self):
        return str(zes_engine_group_v(self.value))


###############################################################################
## @brief Engine group properties
class zes_engine_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("type", zes_engine_group_t),                                   ## [out] The engine group
        ("onSubdevice", ze_bool_t),                                     ## [out] True if this resource is located on a sub-device; false means
                                                                        ## that the resource is on the device of the calling Sysman handle
        ("subdeviceId", c_ulong)                                        ## [out] If onSubdevice is true, this gives the ID of the sub-device
    ]

###############################################################################
## @brief Engine activity counters
## 
## @details
##     - Percent utilization is calculated by taking two snapshots (s1, s2) and
##       using the equation: %util = (s2.activeTime - s1.activeTime) /
##       (s2.timestamp - s1.timestamp)
class zes_engine_stats_t(Structure):
    _fields_ = [
        ("activeTime", c_ulonglong),                                    ## [out] Monotonic counter for time in microseconds that this resource is
                                                                        ## actively running workloads.
        ("timestamp", c_ulonglong)                                      ## [out] Monotonic timestamp counter in microseconds when activeTime
                                                                        ## counter was sampled.
                                                                        ## This timestamp should only be used to calculate delta time between
                                                                        ## snapshots of this structure.
                                                                        ## Never take the delta of this timestamp with the timestamp from a
                                                                        ## different structure since they are not guaranteed to have the same base.
                                                                        ## The absolute value of the timestamp is only valid during within the
                                                                        ## application and may be different on the next execution.
    ]

###############################################################################
## @brief Event types
class zes_event_type_flags_v(IntEnum):
    DEVICE_DETACH = ZE_BIT(0)                       ## Event is triggered when the device is no longer available (due to a
                                                    ## reset or being disabled).
    DEVICE_ATTACH = ZE_BIT(1)                       ## Event is triggered after the device is available again.
    DEVICE_SLEEP_STATE_ENTER = ZE_BIT(2)            ## Event is triggered when the driver is about to put the device into a
                                                    ## deep sleep state
    DEVICE_SLEEP_STATE_EXIT = ZE_BIT(3)             ## Event is triggered when the driver is waking the device up from a deep
                                                    ## sleep state
    FREQ_THROTTLED = ZE_BIT(4)                      ## Event is triggered when the frequency starts being throttled
    ENERGY_THRESHOLD_CROSSED = ZE_BIT(5)            ## Event is triggered when the energy consumption threshold is reached
                                                    ## (use ::zesPowerSetEnergyThreshold() to configure).
    TEMP_CRITICAL = ZE_BIT(6)                       ## Event is triggered when the critical temperature is reached (use
                                                    ## ::zesTemperatureSetConfig() to configure - disabled by default).
    TEMP_THRESHOLD1 = ZE_BIT(7)                     ## Event is triggered when the temperature crosses threshold 1 (use
                                                    ## ::zesTemperatureSetConfig() to configure - disabled by default).
    TEMP_THRESHOLD2 = ZE_BIT(8)                     ## Event is triggered when the temperature crosses threshold 2 (use
                                                    ## ::zesTemperatureSetConfig() to configure - disabled by default).
    MEM_HEALTH = ZE_BIT(9)                          ## Event is triggered when the health of device memory changes.
    FABRIC_PORT_HEALTH = ZE_BIT(10)                 ## Event is triggered when the health of fabric ports change.
    PCI_LINK_HEALTH = ZE_BIT(11)                    ## Event is triggered when the health of the PCI link changes.
    RAS_CORRECTABLE_ERRORS = ZE_BIT(12)             ## Event is triggered when accelerator RAS correctable errors cross
                                                    ## thresholds (use ::zesRasSetConfig() to configure - disabled by
                                                    ## default).
    RAS_UNCORRECTABLE_ERRORS = ZE_BIT(13)           ## Event is triggered when accelerator RAS uncorrectable errors cross
                                                    ## thresholds (use ::zesRasSetConfig() to configure - disabled by
                                                    ## default).
    DEVICE_RESET_REQUIRED = ZE_BIT(14)              ## Event is triggered when the device needs to be reset (use
                                                    ## ::zesDeviceGetState() to determine the reasons for the reset).

class zes_event_type_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Maximum Fabric port model string size
ZES_MAX_FABRIC_PORT_MODEL_SIZE = 256

###############################################################################
## @brief Maximum size of the buffer that will return information about link
##        types
ZES_MAX_FABRIC_LINK_TYPE_SIZE = 256

###############################################################################
## @brief Fabric port status
class zes_fabric_port_status_v(IntEnum):
    UNKNOWN = 0                                     ## The port status cannot be determined
    HEALTHY = 1                                     ## The port is up and operating as expected
    DEGRADED = 2                                    ## The port is up but has quality and/or speed degradation
    FAILED = 3                                      ## Port connection instabilities are preventing workloads making forward
                                                    ## progress
    DISABLED = 4                                    ## The port is configured down

class zes_fabric_port_status_t(c_int):
    def __str__(self):
        return str(zes_fabric_port_status_v(self.value))


###############################################################################
## @brief Fabric port quality degradation reasons
class zes_fabric_port_qual_issue_flags_v(IntEnum):
    LINK_ERRORS = ZE_BIT(0)                         ## Excessive link errors are occurring
    SPEED = ZE_BIT(1)                               ## There is a degradation in the bitrate and/or width of the link

class zes_fabric_port_qual_issue_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Fabric port failure reasons
class zes_fabric_port_failure_flags_v(IntEnum):
    FAILED = ZE_BIT(0)                              ## A previously operating link has failed. Hardware will automatically
                                                    ## retrain this port. This state will persist until either the physical
                                                    ## connection is removed or the link trains successfully.
    TRAINING_TIMEOUT = ZE_BIT(1)                    ## A connection has not been established within an expected time.
                                                    ## Hardware will continue to attempt port training. This status will
                                                    ## persist until either the physical connection is removed or the link
                                                    ## successfully trains.
    FLAPPING = ZE_BIT(2)                            ## Port has excessively trained and then transitioned down for some
                                                    ## period of time. Driver will allow port to continue to train, but will
                                                    ## not enable the port for use until the port has been disabled and
                                                    ## subsequently re-enabled using ::zesFabricPortSetConfig().

class zes_fabric_port_failure_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Unique identifier for a fabric port
## 
## @details
##     - This not a universal identifier. The identified is garanteed to be
##       unique for the current hardware configuration of the system. Changes
##       in the hardware may result in a different identifier for a given port.
##     - The main purpose of this identifier to build up an instantaneous
##       topology map of system connectivity. An application should enumerate
##       all fabric ports and match ::zes_fabric_port_state_t.remotePortId to
##       ::zes_fabric_port_properties_t.portId.
class zes_fabric_port_id_t(Structure):
    _fields_ = [
        ("fabricId", c_ulong),                                          ## [out] Unique identifier for the fabric end-point
        ("attachId", c_ulong),                                          ## [out] Unique identifier for the device attachment point
        ("portNumber", c_ubyte)                                         ## [out] The logical port number (this is typically marked somewhere on
                                                                        ## the physical device)
    ]

###############################################################################
## @brief Fabric port speed in one direction
class zes_fabric_port_speed_t(Structure):
    _fields_ = [
        ("bitRate", c_int64_t),                                         ## [out] Bits/sec that the link is operating at. A value of -1 means that
                                                                        ## this property is unknown.
        ("width", c_int32_t)                                            ## [out] The number of lanes. A value of -1 means that this property is
                                                                        ## unknown.
    ]

###############################################################################
## @brief Fabric port properties
class zes_fabric_port_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("model", c_char * ZES_MAX_FABRIC_PORT_MODEL_SIZE),             ## [out] Description of port technology. Will be set to the string
                                                                        ## "unkown" if this cannot be determined for this port.
        ("onSubdevice", ze_bool_t),                                     ## [out] True if the port is located on a sub-device; false means that
                                                                        ## the port is on the device of the calling Sysman handle
        ("subdeviceId", c_ulong),                                       ## [out] If onSubdevice is true, this gives the ID of the sub-device
        ("portId", zes_fabric_port_id_t),                               ## [out] The unique port identifier
        ("maxRxSpeed", zes_fabric_port_speed_t),                        ## [out] Maximum speed supported by the receive side of the port (sum of
                                                                        ## all lanes)
        ("maxTxSpeed", zes_fabric_port_speed_t)                         ## [out] Maximum speed supported by the transmit side of the port (sum of
                                                                        ## all lanes)
    ]

###############################################################################
## @brief Provides information about the fabric link attached to a port
class zes_fabric_link_type_t(Structure):
    _fields_ = [
        ("desc", c_char * ZES_MAX_FABRIC_LINK_TYPE_SIZE)                ## [out] This provides a static textural description of the physic
                                                                        ## attachment type. Will be set to the string "unkown" if this cannot be
                                                                        ## determined for this port.
    ]

###############################################################################
## @brief Fabric port configuration
class zes_fabric_port_config_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("enabled", ze_bool_t),                                         ## [in,out] Port is configured up/down
        ("beaconing", ze_bool_t)                                        ## [in,out] Beaconing is configured on/off
    ]

###############################################################################
## @brief Fabric port state
class zes_fabric_port_state_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("status", zes_fabric_port_status_t),                           ## [out] The current status of the port
        ("qualityIssues", zes_fabric_port_qual_issue_flags_t),          ## [out] If status is ::ZES_FABRIC_PORT_STATUS_DEGRADED, 
                                                                        ## then this gives a combination of ::zes_fabric_port_qual_issue_flag_t
                                                                        ## for quality issues that have been detected;
                                                                        ## otherwise, 0 indicates there are no quality issues with the link at
                                                                        ## this time.
        ("failureReasons", zes_fabric_port_failure_flags_t),            ## [out] If status is ::ZES_FABRIC_PORT_STATUS_FAILED,
                                                                        ## then this gives a combination of ::zes_fabric_port_failure_flag_t for
                                                                        ## reasons for the connection instability;
                                                                        ## otherwise, 0 indicates there are no connection stability issues at
                                                                        ## this time.
        ("remotePortId", zes_fabric_port_id_t),                         ## [out] The unique port identifier for the remote connection point if
                                                                        ## status is ::ZES_FABRIC_PORT_STATUS_HEALTHY,
                                                                        ## ::ZES_FABRIC_PORT_STATUS_DEGRADED or ::ZES_FABRIC_PORT_STATUS_FAILED
        ("rxSpeed", zes_fabric_port_speed_t),                           ## [out] Current maximum receive speed (sum of all lanes)
        ("txSpeed", zes_fabric_port_speed_t)                            ## [out] Current maximum transmit speed (sum of all lanes)
    ]

###############################################################################
## @brief Fabric port throughput.
class zes_fabric_port_throughput_t(Structure):
    _fields_ = [
        ("timestamp", c_ulonglong),                                     ## [out] Monotonic timestamp counter in microseconds when the measurement
                                                                        ## was made.
                                                                        ## This timestamp should only be used to calculate delta time between
                                                                        ## snapshots of this structure.
                                                                        ## Never take the delta of this timestamp with the timestamp from a
                                                                        ## different structure since they are not guaranteed to have the same base.
                                                                        ## The absolute value of the timestamp is only valid during within the
                                                                        ## application and may be different on the next execution.
        ("rxCounter", c_ulonglong),                                     ## [out] Monotonic counter for the number of bytes received (sum of all
                                                                        ## lanes). This includes all protocol overhead, not only the GPU traffic.
        ("txCounter", c_ulonglong)                                      ## [out] Monotonic counter for the number of bytes transmitted (sum of
                                                                        ## all lanes). This includes all protocol overhead, not only the GPU
                                                                        ## traffic.
    ]

###############################################################################
## @brief Fan resource speed mode
class zes_fan_speed_mode_v(IntEnum):
    DEFAULT = 0                                     ## The fan speed is operating using the hardware default settings
    FIXED = 1                                       ## The fan speed is currently set to a fixed value
    TABLE = 2                                       ## The fan speed is currently controlled dynamically by hardware based on
                                                    ## a temp/speed table

class zes_fan_speed_mode_t(c_int):
    def __str__(self):
        return str(zes_fan_speed_mode_v(self.value))


###############################################################################
## @brief Fan speed units
class zes_fan_speed_units_v(IntEnum):
    RPM = 0                                         ## The fan speed is in units of revolutions per minute (rpm)
    PERCENT = 1                                     ## The fan speed is a percentage of the maximum speed of the fan

class zes_fan_speed_units_t(c_int):
    def __str__(self):
        return str(zes_fan_speed_units_v(self.value))


###############################################################################
## @brief Fan speed
class zes_fan_speed_t(Structure):
    _fields_ = [
        ("speed", c_int32_t),                                           ## [in,out] The speed of the fan. On output, a value of -1 indicates that
                                                                        ## there is no fixed fan speed setting.
        ("units", zes_fan_speed_units_t)                                ## [in,out] The units that the fan speed is expressed in. On output, if
                                                                        ## fan speed is -1 then units should be ignored.
    ]

###############################################################################
## @brief Fan temperature/speed pair
class zes_fan_temp_speed_t(Structure):
    _fields_ = [
        ("temperature", c_ulong),                                       ## [in,out] Temperature in degrees Celsius.
        ("speed", zes_fan_speed_t)                                      ## [in,out] The speed of the fan
    ]

###############################################################################
## @brief Maximum number of fan temperature/speed pairs in the fan speed table.
ZES_FAN_TEMP_SPEED_PAIR_COUNT = 32

###############################################################################
## @brief Fan speed table
class zes_fan_speed_table_t(Structure):
    _fields_ = [
        ("numPoints", c_int32_t),                                       ## [in,out] The number of valid points in the fan speed table. 0 means
                                                                        ## that there is no fan speed table configured. -1 means that a fan speed
                                                                        ## table is not supported by the hardware.
        ("table", zes_fan_temp_speed_t * ZES_FAN_TEMP_SPEED_PAIR_COUNT) ## [in,out] Array of temperature/fan speed pairs. The table is ordered
                                                                        ## based on temperature from lowest to highest.
    ]

###############################################################################
## @brief Fan properties
class zes_fan_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("onSubdevice", ze_bool_t),                                     ## [out] True if the resource is located on a sub-device; false means
                                                                        ## that the resource is on the device of the calling Sysman handle
        ("subdeviceId", c_ulong),                                       ## [out] If onSubdevice is true, this gives the ID of the sub-device
        ("canControl", ze_bool_t),                                      ## [out] Indicates if software can control the fan speed assuming the
                                                                        ## user has permissions
        ("supportedModes", c_ulong),                                    ## [out] Bitfield of supported fan configuration modes
                                                                        ## (1<<::zes_fan_speed_mode_t)
        ("supportedUnits", c_ulong),                                    ## [out] Bitfield of supported fan speed units
                                                                        ## (1<<::zes_fan_speed_units_t)
        ("maxRPM", c_int32_t),                                          ## [out] The maximum RPM of the fan. A value of -1 means that this
                                                                        ## property is unknown. 
        ("maxPoints", c_int32_t)                                        ## [out] The maximum number of points in the fan temp/speed table. A
                                                                        ## value of -1 means that this fan doesn't support providing a temp/speed
                                                                        ## table.
    ]

###############################################################################
## @brief Fan configuration
class zes_fan_config_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("mode", zes_fan_speed_mode_t),                                 ## [in,out] The fan speed mode (fixed, temp-speed table)
        ("speedFixed", zes_fan_speed_t),                                ## [in,out] The current fixed fan speed setting
        ("speedTable", zes_fan_speed_table_t)                           ## [out] A table containing temperature/speed pairs
    ]

###############################################################################
## @brief Firmware properties
class zes_firmware_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("onSubdevice", ze_bool_t),                                     ## [out] True if the resource is located on a sub-device; false means
                                                                        ## that the resource is on the device of the calling Sysman handle
        ("subdeviceId", c_ulong),                                       ## [out] If onSubdevice is true, this gives the ID of the sub-device
        ("canControl", ze_bool_t),                                      ## [out] Indicates if software can flash the firmware assuming the user
                                                                        ## has permissions
        ("name", c_char * ZES_STRING_PROPERTY_SIZE),                    ## [out] NULL terminated string value. The string "unknown" will be
                                                                        ## returned if this property cannot be determined.
        ("version", c_char * ZES_STRING_PROPERTY_SIZE)                  ## [out] NULL terminated string value. The string "unknown" will be
                                                                        ## returned if this property cannot be determined.
    ]

###############################################################################
## @brief Frequency domains.
class zes_freq_domain_v(IntEnum):
    GPU = 0                                         ## GPU Core Domain.
    MEMORY = 1                                      ## Local Memory Domain.

class zes_freq_domain_t(c_int):
    def __str__(self):
        return str(zes_freq_domain_v(self.value))


###############################################################################
## @brief Frequency properties
## 
## @details
##     - Indicates if this frequency domain can be overclocked (if true,
##       functions such as ::zesFrequencyOcSetFrequencyTarget() are supported).
##     - The min/max hardware frequencies are specified for non-overclock
##       configurations. For overclock configurations, use
##       ::zesFrequencyOcGetFrequencyTarget() to determine the maximum
##       frequency that can be requested.
class zes_freq_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("type", zes_freq_domain_t),                                    ## [out] The hardware block that this frequency domain controls (GPU,
                                                                        ## memory, ...)
        ("onSubdevice", ze_bool_t),                                     ## [out] True if this resource is located on a sub-device; false means
                                                                        ## that the resource is on the device of the calling Sysman handle
        ("subdeviceId", c_ulong),                                       ## [out] If onSubdevice is true, this gives the ID of the sub-device
        ("canControl", ze_bool_t),                                      ## [out] Indicates if software can control the frequency of this domain
                                                                        ## assuming the user has permissions
        ("isThrottleEventSupported", ze_bool_t),                        ## [out] Indicates if software can register to receive event
                                                                        ## ::ZES_EVENT_TYPE_FLAG_FREQ_THROTTLED
        ("min", c_double),                                              ## [out] The minimum hardware clock frequency in units of MHz.
        ("max", c_double)                                               ## [out] The maximum non-overclock hardware clock frequency in units of
                                                                        ## MHz.
    ]

###############################################################################
## @brief Frequency range between which the hardware can operate. The limits can
##        be above or below the hardware limits - the hardware will clamp
##        appropriately.
class zes_freq_range_t(Structure):
    _fields_ = [
        ("min", c_double),                                              ## [in,out] The min frequency in MHz below which hardware frequency
                                                                        ## management will not request frequencies. On input, setting to 0 will
                                                                        ## permit the frequency to go down to the hardware minimum. On output, a
                                                                        ## negative value indicates that no external minimum frequency limit is
                                                                        ## in effect.
        ("max", c_double)                                               ## [in,out] The max frequency in MHz above which hardware frequency
                                                                        ## management will not request frequencies. On input, setting to 0 or a
                                                                        ## very big number will permit the frequency to go all the way up to the
                                                                        ## hardware maximum. On output, a negative number indicates that no
                                                                        ## external maximum frequency limit is in effect.
    ]

###############################################################################
## @brief Frequency throttle reasons
class zes_freq_throttle_reason_flags_v(IntEnum):
    AVE_PWR_CAP = ZE_BIT(0)                         ## frequency throttled due to average power excursion (PL1)
    BURST_PWR_CAP = ZE_BIT(1)                       ## frequency throttled due to burst power excursion (PL2)
    CURRENT_LIMIT = ZE_BIT(2)                       ## frequency throttled due to current excursion (PL4)
    THERMAL_LIMIT = ZE_BIT(3)                       ## frequency throttled due to thermal excursion (T > TjMax)
    PSU_ALERT = ZE_BIT(4)                           ## frequency throttled due to power supply assertion
    SW_RANGE = ZE_BIT(5)                            ## frequency throttled due to software supplied frequency range
    HW_RANGE = ZE_BIT(6)                            ## frequency throttled due to a sub block that has a lower frequency
                                                    ## range when it receives clocks

class zes_freq_throttle_reason_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Frequency state
class zes_freq_state_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("currentVoltage", c_double),                                   ## [out] Current voltage in Volts. A negative value indicates that this
                                                                        ## property is not known.
        ("request", c_double),                                          ## [out] The current frequency request in MHz. A negative value indicates
                                                                        ## that this property is not known.
        ("tdp", c_double),                                              ## [out] The maximum frequency in MHz supported under the current TDP
                                                                        ## conditions. This fluctuates dynamically based on the power and thermal
                                                                        ## limits of the part. A negative value indicates that this property is
                                                                        ## not known.
        ("efficient", c_double),                                        ## [out] The efficient minimum frequency in MHz. A negative value
                                                                        ## indicates that this property is not known.
        ("actual", c_double),                                           ## [out] The resolved frequency in MHz. A negative value indicates that
                                                                        ## this property is not known.
        ("throttleReasons", zes_freq_throttle_reason_flags_t)           ## [out] The reasons that the frequency is being limited by the hardware.
                                                                        ## Returns 0 (frequency not throttled) or a combination of ::zes_freq_throttle_reason_flag_t.
    ]

###############################################################################
## @brief Frequency throttle time snapshot
## 
## @details
##     - Percent time throttled is calculated by taking two snapshots (s1, s2)
##       and using the equation: %throttled = (s2.throttleTime -
##       s1.throttleTime) / (s2.timestamp - s1.timestamp)
class zes_freq_throttle_time_t(Structure):
    _fields_ = [
        ("throttleTime", c_ulonglong),                                  ## [out] The monotonic counter of time in microseconds that the frequency
                                                                        ## has been limited by the hardware.
        ("timestamp", c_ulonglong)                                      ## [out] Microsecond timestamp when throttleTime was captured.
                                                                        ## This timestamp should only be used to calculate delta time between
                                                                        ## snapshots of this structure.
                                                                        ## Never take the delta of this timestamp with the timestamp from a
                                                                        ## different structure since they are not guaranteed to have the same base.
                                                                        ## The absolute value of the timestamp is only valid during within the
                                                                        ## application and may be different on the next execution.
    ]

###############################################################################
## @brief Overclocking modes
class zes_oc_mode_v(IntEnum):
    OFF = 0                                         ## Overclocking if off - hardware is running using factory default
                                                    ## voltages/frequencies.
    OVERRIDE = 1                                    ## Overclock override mode - In this mode, a fixed user-supplied voltage
                                                    ## is applied independent of the frequency request. The maximum permitted
                                                    ## frequency can also be increased. This mode disables INTERPOLATIVE and
                                                    ## FIXED modes.
    INTERPOLATIVE = 2                               ## Overclock interpolative mode - In this mode, the voltage/frequency
                                                    ## curve can be extended with a new voltage/frequency point that will be
                                                    ## interpolated. The existing voltage/frequency points can also be offset
                                                    ## (up or down) by a fixed voltage. This mode disables FIXED and OVERRIDE
                                                    ## modes.
    FIXED = 3                                       ## Overclocking fixed Mode - In this mode, hardware will disable most
                                                    ## frequency throttling and lock the frequency and voltage at the
                                                    ## specified overclock values. This mode disables OVERRIDE and
                                                    ## INTERPOLATIVE modes. This mode can damage the part, most of the
                                                    ## protections are disabled on this mode.

class zes_oc_mode_t(c_int):
    def __str__(self):
        return str(zes_oc_mode_v(self.value))


###############################################################################
## @brief Overclocking properties
## 
## @details
##     - Provides all the overclocking capabilities and properties supported by
##       the device for the frequency domain.
class zes_oc_capabilities_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("isOcSupported", ze_bool_t),                                   ## [out] Indicates if any overclocking features are supported on this
                                                                        ## frequency domain.
        ("maxFactoryDefaultFrequency", c_double),                       ## [out] Factory default non-overclock maximum frequency in Mhz.
        ("maxFactoryDefaultVoltage", c_double),                         ## [out] Factory default voltage used for the non-overclock maximum
                                                                        ## frequency in MHz.
        ("maxOcFrequency", c_double),                                   ## [out] Maximum hardware overclocking frequency limit in Mhz.
        ("minOcVoltageOffset", c_double),                               ## [out] The minimum voltage offset that can be applied to the
                                                                        ## voltage/frequency curve. Note that this number can be negative.
        ("maxOcVoltageOffset", c_double),                               ## [out] The maximum voltage offset that can be applied to the
                                                                        ## voltage/frequency curve.
        ("maxOcVoltage", c_double),                                     ## [out] The maximum overclock voltage that hardware supports.
        ("isTjMaxSupported", ze_bool_t),                                ## [out] Indicates if the maximum temperature limit (TjMax) can be
                                                                        ## changed for this frequency domain.
        ("isIccMaxSupported", ze_bool_t),                               ## [out] Indicates if the maximum current (IccMax) can be changed for
                                                                        ## this frequency domain.
        ("isHighVoltModeCapable", ze_bool_t),                           ## [out] Indicates if this frequency domains supports a feature to set
                                                                        ## very high voltages.
        ("isHighVoltModeEnabled", ze_bool_t),                           ## [out] Indicates if very high voltages are permitted on this frequency
                                                                        ## domain.
        ("isExtendedModeSupported", ze_bool_t),                         ## [out] Indicates if the extended overclocking features are supported.
                                                                        ## If this is supported, increments are on 1 Mhz basis.
        ("isFixedModeSupported", ze_bool_t)                             ## [out] Indicates if the fixed mode is supported. In this mode, hardware
                                                                        ## will disable most frequency throttling and lock the frequency and
                                                                        ## voltage at the specified overclock values.
    ]

###############################################################################
## @brief LED properties
class zes_led_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("onSubdevice", ze_bool_t),                                     ## [out] True if the resource is located on a sub-device; false means
                                                                        ## that the resource is on the device of the calling Sysman handle
        ("subdeviceId", c_ulong),                                       ## [out] If onSubdevice is true, this gives the ID of the sub-device
        ("canControl", ze_bool_t),                                      ## [out] Indicates if software can control the LED assuming the user has
                                                                        ## permissions
        ("haveRGB", ze_bool_t)                                          ## [out] Indicates if the LED is RGB capable
    ]

###############################################################################
## @brief LED color
class zes_led_color_t(Structure):
    _fields_ = [
        ("red", c_double),                                              ## [in,out][range(0.0, 1.0)] The LED red value. On output, a value less
                                                                        ## than 0.0 indicates that the color is not known.
        ("green", c_double),                                            ## [in,out][range(0.0, 1.0)] The LED green value. On output, a value less
                                                                        ## than 0.0 indicates that the color is not known.
        ("blue", c_double)                                              ## [in,out][range(0.0, 1.0)] The LED blue value. On output, a value less
                                                                        ## than 0.0 indicates that the color is not known.
    ]

###############################################################################
## @brief LED state
class zes_led_state_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("isOn", ze_bool_t),                                            ## [out] Indicates if the LED is on or off
        ("color", zes_led_color_t)                                      ## [out] Color of the LED
    ]

###############################################################################
## @brief Memory module types
class zes_mem_type_v(IntEnum):
    HBM = 0                                         ## HBM memory
    DDR = 1                                         ## DDR memory
    DDR3 = 2                                        ## DDR3 memory
    DDR4 = 3                                        ## DDR4 memory
    DDR5 = 4                                        ## DDR5 memory
    LPDDR = 5                                       ## LPDDR memory
    LPDDR3 = 6                                      ## LPDDR3 memory
    LPDDR4 = 7                                      ## LPDDR4 memory
    LPDDR5 = 8                                      ## LPDDR5 memory
    SRAM = 9                                        ## SRAM memory
    L1 = 10                                         ## L1 cache
    L3 = 11                                         ## L3 cache
    GRF = 12                                        ## Execution unit register file
    SLM = 13                                        ## Execution unit shared local memory
    GDDR4 = 14                                      ## GDDR4 memory
    GDDR5 = 15                                      ## GDDR5 memory
    GDDR5X = 16                                     ## GDDR5X memory
    GDDR6 = 17                                      ## GDDR6 memory
    GDDR6X = 18                                     ## GDDR6X memory
    GDDR7 = 19                                      ## GDDR7 memory

class zes_mem_type_t(c_int):
    def __str__(self):
        return str(zes_mem_type_v(self.value))


###############################################################################
## @brief Memory module location
class zes_mem_loc_v(IntEnum):
    SYSTEM = 0                                      ## System memory
    DEVICE = 1                                      ## On board local device memory

class zes_mem_loc_t(c_int):
    def __str__(self):
        return str(zes_mem_loc_v(self.value))


###############################################################################
## @brief Memory health
class zes_mem_health_v(IntEnum):
    UNKNOWN = 0                                     ## The memory health cannot be determined.
    OK = 1                                          ## All memory channels are healthy.
    DEGRADED = 2                                    ## Excessive correctable errors have been detected on one or more
                                                    ## channels. Device should be reset.
    CRITICAL = 3                                    ## Operating with reduced memory to cover banks with too many
                                                    ## uncorrectable errors.
    REPLACE = 4                                     ## Device should be replaced due to excessive uncorrectable errors.

class zes_mem_health_t(c_int):
    def __str__(self):
        return str(zes_mem_health_v(self.value))


###############################################################################
## @brief Memory properties
class zes_mem_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("type", zes_mem_type_t),                                       ## [out] The memory type
        ("onSubdevice", ze_bool_t),                                     ## [out] True if this resource is located on a sub-device; false means
                                                                        ## that the resource is on the device of the calling Sysman handle
        ("subdeviceId", c_ulong),                                       ## [out] If onSubdevice is true, this gives the ID of the sub-device
        ("location", zes_mem_loc_t),                                    ## [out] Location of this memory (system, device)
        ("physicalSize", c_ulonglong),                                  ## [out] Physical memory size in bytes. A value of 0 indicates that this
                                                                        ## property is not known. However, a call to ::zesMemoryGetState() will
                                                                        ## correctly return the total size of usable memory.
        ("busWidth", c_int32_t),                                        ## [out] Width of the memory bus. A value of -1 means that this property
                                                                        ## is unknown.
        ("numChannels", c_int32_t)                                      ## [out] The number of memory channels. A value of -1 means that this
                                                                        ## property is unknown.
    ]

###############################################################################
## @brief Memory state - health, allocated
## 
## @details
##     - Percent allocation is given by 100 * (size - free / size.
##     - Percent free is given by 100 * free / size.
class zes_mem_state_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("health", zes_mem_health_t),                                   ## [out] Indicates the health of the memory
        ("free", c_ulonglong),                                          ## [out] The free memory in bytes
        ("size", c_ulonglong)                                           ## [out] The total allocatable memory in bytes (can be less than
                                                                        ## ::zes_mem_properties_t.physicalSize)
    ]

###############################################################################
## @brief Memory bandwidth
## 
## @details
##     - Percent bandwidth is calculated by taking two snapshots (s1, s2) and
##       using the equation: %bw = 10^6 * ((s2.readCounter - s1.readCounter) +
##       (s2.writeCounter - s1.writeCounter)) / (s2.maxBandwidth *
##       (s2.timestamp - s1.timestamp))
class zes_mem_bandwidth_t(Structure):
    _fields_ = [
        ("readCounter", c_ulonglong),                                   ## [out] Total bytes read from memory
        ("writeCounter", c_ulonglong),                                  ## [out] Total bytes written to memory
        ("maxBandwidth", c_ulonglong),                                  ## [out] Current maximum bandwidth in units of bytes/sec
        ("timestamp", c_ulonglong)                                      ## [out] The timestamp when these measurements were sampled.
                                                                        ## This timestamp should only be used to calculate delta time between
                                                                        ## snapshots of this structure.
                                                                        ## Never take the delta of this timestamp with the timestamp from a
                                                                        ## different structure since they are not guaranteed to have the same base.
                                                                        ## The absolute value of the timestamp is only valid during within the
                                                                        ## application and may be different on the next execution.
    ]

###############################################################################
## @brief Static information about a Performance Factor domain
class zes_perf_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("onSubdevice", ze_bool_t),                                     ## [out] True if this Performance Factor affects accelerators located on
                                                                        ## a sub-device
        ("subdeviceId", c_ulong),                                       ## [out] If onSubdevice is true, this gives the ID of the sub-device
        ("engines", zes_engine_type_flags_t)                            ## [out] Bitfield of accelerator engine types that are affected by this
                                                                        ## Performance Factor.
    ]

###############################################################################
## @brief Properties related to device power settings
class zes_power_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("onSubdevice", ze_bool_t),                                     ## [out] True if this resource is located on a sub-device; false means
                                                                        ## that the resource is on the device of the calling Sysman handle
        ("subdeviceId", c_ulong),                                       ## [out] If onSubdevice is true, this gives the ID of the sub-device
        ("canControl", ze_bool_t),                                      ## [out] Software can change the power limits of this domain assuming the
                                                                        ## user has permissions.
        ("isEnergyThresholdSupported", ze_bool_t),                      ## [out] Indicates if this power domain supports the energy threshold
                                                                        ## event (::ZES_EVENT_TYPE_FLAG_ENERGY_THRESHOLD_CROSSED).
        ("defaultLimit", c_int32_t),                                    ## [out] The factory default TDP power limit of the part in milliwatts. A
                                                                        ## value of -1 means that this is not known.
        ("minLimit", c_int32_t),                                        ## [out] The minimum power limit in milliwatts that can be requested.
        ("maxLimit", c_int32_t)                                         ## [out] The maximum power limit in milliwatts that can be requested.
    ]

###############################################################################
## @brief Energy counter snapshot
## 
## @details
##     - Average power is calculated by taking two snapshots (s1, s2) and using
##       the equation: PowerWatts = (s2.energy - s1.energy) / (s2.timestamp -
##       s1.timestamp)
class zes_power_energy_counter_t(Structure):
    _fields_ = [
        ("energy", c_ulonglong),                                        ## [out] The monotonic energy counter in microjoules.
        ("timestamp", c_ulonglong)                                      ## [out] Microsecond timestamp when energy was captured.
                                                                        ## This timestamp should only be used to calculate delta time between
                                                                        ## snapshots of this structure.
                                                                        ## Never take the delta of this timestamp with the timestamp from a
                                                                        ## different structure since they are not guaranteed to have the same base.
                                                                        ## The absolute value of the timestamp is only valid during within the
                                                                        ## application and may be different on the next execution.
    ]

###############################################################################
## @brief Sustained power limits
## 
## @details
##     - The power controller (Punit) will throttle the operating frequency if
##       the power averaged over a window (typically seconds) exceeds this
##       limit.
class zes_power_sustained_limit_t(Structure):
    _fields_ = [
        ("enabled", ze_bool_t),                                         ## [in,out] indicates if the limit is enabled (true) or ignored (false)
        ("power", c_int32_t),                                           ## [in,out] power limit in milliwatts
        ("interval", c_int32_t)                                         ## [in,out] power averaging window (Tau) in milliseconds
    ]

###############################################################################
## @brief Burst power limit
## 
## @details
##     - The power controller (Punit) will throttle the operating frequency of
##       the device if the power averaged over a few milliseconds exceeds a
##       limit known as PL2. Typically PL2 > PL1 so that it permits the
##       frequency to burst higher for short periods than would be otherwise
##       permitted by PL1.
class zes_power_burst_limit_t(Structure):
    _fields_ = [
        ("enabled", ze_bool_t),                                         ## [in,out] indicates if the limit is enabled (true) or ignored (false)
        ("power", c_int32_t)                                            ## [in,out] power limit in milliwatts
    ]

###############################################################################
## @brief Peak power limit
## 
## @details
##     - The power controller (Punit) will reactively/proactively throttle the
##       operating frequency of the device when the instantaneous/100usec power
##       exceeds this limit. The limit is known as PL4 or Psys. It expresses
##       the maximum power that can be drawn from the power supply.
##     - If this power limit is removed or set too high, the power supply will
##       generate an interrupt when it detects an overcurrent condition and the
##       power controller will throttle the device frequencies down to min. It
##       is thus better to tune the PL4 value in order to avoid such
##       excursions.
class zes_power_peak_limit_t(Structure):
    _fields_ = [
        ("powerAC", c_int32_t),                                         ## [in,out] power limit in milliwatts for the AC power source.
        ("powerDC", c_int32_t)                                          ## [in,out] power limit in milliwatts for the DC power source. On input,
                                                                        ## this is ignored if the product does not have a battery. On output,
                                                                        ## this will be -1 if the product does not have a battery.
    ]

###############################################################################
## @brief Energy threshold
## 
## @details
##     - .
class zes_energy_threshold_t(Structure):
    _fields_ = [
        ("enable", ze_bool_t),                                          ## [in,out] Indicates if the energy threshold is enabled.
        ("threshold", c_double),                                        ## [in,out] The energy threshold in Joules. Will be 0.0 if no threshold
                                                                        ## has been set.
        ("processId", c_ulong)                                          ## [in,out] The host process ID that set the energy threshold. Will be
                                                                        ## 0xFFFFFFFF if no threshold has been set.
    ]

###############################################################################
## @brief PSU voltage status
class zes_psu_voltage_status_v(IntEnum):
    UNKNOWN = 0                                     ## The status of the power supply voltage controllers cannot be
                                                    ## determined
    NORMAL = 1                                      ## No unusual voltages have been detected
    OVER = 2                                        ## Over-voltage has occurred
    UNDER = 3                                       ## Under-voltage has occurred

class zes_psu_voltage_status_t(c_int):
    def __str__(self):
        return str(zes_psu_voltage_status_v(self.value))


###############################################################################
## @brief Static properties of the power supply
class zes_psu_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("onSubdevice", ze_bool_t),                                     ## [out] True if the resource is located on a sub-device; false means
                                                                        ## that the resource is on the device of the calling Sysman handle
        ("subdeviceId", c_ulong),                                       ## [out] If onSubdevice is true, this gives the ID of the sub-device
        ("haveFan", ze_bool_t),                                         ## [out] True if the power supply has a fan
        ("ampLimit", c_int32_t)                                         ## [out] The maximum electrical current in milliamperes that can be
                                                                        ## drawn. A value of -1 indicates that this property cannot be
                                                                        ## determined.
    ]

###############################################################################
## @brief Dynamic state of the power supply
class zes_psu_state_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("voltStatus", zes_psu_voltage_status_t),                       ## [out] The current PSU voltage status
        ("fanFailed", ze_bool_t),                                       ## [out] Indicates if the fan has failed
        ("temperature", c_int32_t),                                     ## [out] Read the current heatsink temperature in degrees Celsius. A
                                                                        ## value of -1 indicates that this property cannot be determined.
        ("current", c_int32_t)                                          ## [out] The amps being drawn in milliamperes. A value of -1 indicates
                                                                        ## that this property cannot be determined.
    ]

###############################################################################
## @brief RAS error type
class zes_ras_error_type_v(IntEnum):
    CORRECTABLE = 0                                 ## Errors were corrected by hardware
    UNCORRECTABLE = 1                               ## Error were not corrected

class zes_ras_error_type_t(c_int):
    def __str__(self):
        return str(zes_ras_error_type_v(self.value))


###############################################################################
## @brief RAS error categories
class zes_ras_error_cat_v(IntEnum):
    RESET = 0                                       ## The number of accelerator engine resets attempted by the driver
    PROGRAMMING_ERRORS = 1                          ## The number of hardware exceptions generated by the way workloads have
                                                    ## programmed the hardware
    DRIVER_ERRORS = 2                               ## The number of low level driver communication errors have occurred
    COMPUTE_ERRORS = 3                              ## The number of errors that have occurred in the compute accelerator
                                                    ## hardware
    NON_COMPUTE_ERRORS = 4                          ## The number of errors that have occurred in the fixed-function
                                                    ## accelerator hardware
    CACHE_ERRORS = 5                                ## The number of errors that have occurred in caches (L1/L3/register
                                                    ## file/shared local memory/sampler)
    DISPLAY_ERRORS = 6                              ## The number of errors that have occurred in the display

class zes_ras_error_cat_t(c_int):
    def __str__(self):
        return str(zes_ras_error_cat_v(self.value))


###############################################################################
## @brief The maximum number of categories
ZES_MAX_RAS_ERROR_CATEGORY_COUNT = 7

###############################################################################
## @brief RAS properties
class zes_ras_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("type", zes_ras_error_type_t),                                 ## [out] The type of RAS error
        ("onSubdevice", ze_bool_t),                                     ## [out] True if the resource is located on a sub-device; false means
                                                                        ## that the resource is on the device of the calling Sysman handle
        ("subdeviceId", c_ulong)                                        ## [out] If onSubdevice is true, this gives the ID of the sub-device
    ]

###############################################################################
## @brief RAS error details
class zes_ras_state_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("category", c_ulonglong * ZES_MAX_RAS_ERROR_CATEGORY_COUNT)    ## [in][out] Breakdown of error by category
    ]

###############################################################################
## @brief RAS error configuration - thresholds used for triggering RAS events
##        (::ZES_EVENT_TYPE_FLAG_RAS_CORRECTABLE_ERRORS,
##        ::ZES_EVENT_TYPE_FLAG_RAS_UNCORRECTABLE_ERRORS)
## 
## @details
##     - The driver maintains a total counter which is updated every time a
##       hardware block covered by the corresponding RAS error set notifies
##       that an error has occurred. When this total count goes above the
##       totalThreshold specified below, a RAS event is triggered.
##     - The driver also maintains a counter for each category of RAS error
##       (see ::zes_ras_state_t for a breakdown). Each time a hardware block of
##       that category notifies that an error has occurred, that corresponding
##       category counter is updated. When it goes above the threshold
##       specified in detailedThresholds, a RAS event is triggered.
class zes_ras_config_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("totalThreshold", c_ulonglong),                                ## [in,out] If the total RAS errors exceeds this threshold, the event
                                                                        ## will be triggered. A value of 0ULL disables triggering the event based
                                                                        ## on the total counter.
        ("detailedThresholds", zes_ras_state_t)                         ## [in,out] If the RAS errors for each category exceed the threshold for
                                                                        ## that category, the event will be triggered. A value of 0ULL will
                                                                        ## disable an event being triggered for that category.
    ]

###############################################################################
## @brief Scheduler mode
class zes_sched_mode_v(IntEnum):
    TIMEOUT = 0                                     ## Multiple applications or contexts are submitting work to the hardware.
                                                    ## When higher priority work arrives, the scheduler attempts to pause the
                                                    ## current executing work within some timeout interval, then submits the
                                                    ## other work.
    TIMESLICE = 1                                   ## The scheduler attempts to fairly timeslice hardware execution time
                                                    ## between multiple contexts submitting work to the hardware
                                                    ## concurrently.
    EXCLUSIVE = 2                                   ## Any application or context can run indefinitely on the hardware
                                                    ## without being preempted or terminated. All pending work for other
                                                    ## contexts must wait until the running context completes with no further
                                                    ## submitted work.
    COMPUTE_UNIT_DEBUG = 3                          ## This is a special mode that must ben enabled when debugging an
                                                    ## application that uses this device e.g. using the Level0 Debug API. It
                                                    ## has the effect of disabling any timeouts on workload execution time
                                                    ## and will change workload scheduling to ensure debug accuracy.

class zes_sched_mode_t(c_int):
    def __str__(self):
        return str(zes_sched_mode_v(self.value))


###############################################################################
## @brief Properties related to scheduler component
class zes_sched_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("onSubdevice", ze_bool_t),                                     ## [out] True if this resource is located on a sub-device; false means
                                                                        ## that the resource is on the device of the calling Sysman handle
        ("subdeviceId", c_ulong),                                       ## [out] If onSubdevice is true, this gives the ID of the sub-device
        ("canControl", ze_bool_t),                                      ## [out] Software can change the scheduler component configuration
                                                                        ## assuming the user has permissions.
        ("engines", zes_engine_type_flags_t),                           ## [out] Bitfield of accelerator engine types that are managed by this
                                                                        ## scheduler component. Note that there can be more than one scheduler
                                                                        ## component for the same type of accelerator engine.
        ("supportedModes", c_ulong)                                     ## [out] Bitfield of scheduler modes that can be configured for this
                                                                        ## scheduler component (bitfield of 1<<::zes_sched_mode_t).
    ]

###############################################################################
## @brief Disable forward progress guard timeout.
ZES_SCHED_WATCHDOG_DISABLE = (~(0ULL))

###############################################################################
## @brief Configuration for timeout scheduler mode (::ZES_SCHED_MODE_TIMEOUT)
class zes_sched_timeout_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("watchdogTimeout", c_ulonglong)                                ## [in,out] The maximum time in microseconds that the scheduler will wait
                                                                        ## for a batch of work submitted to a hardware engine to complete or to
                                                                        ## be preempted so as to run another context.
                                                                        ## If this time is exceeded, the hardware engine is reset and the context terminated.
                                                                        ## If set to ::ZES_SCHED_WATCHDOG_DISABLE, a running workload can run as
                                                                        ## long as it wants without being terminated, but preemption attempts to
                                                                        ## run other contexts are permitted but not enforced.
    ]

###############################################################################
## @brief Configuration for timeslice scheduler mode
##        (::ZES_SCHED_MODE_TIMESLICE)
class zes_sched_timeslice_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("interval", c_ulonglong),                                      ## [in,out] The average interval in microseconds that a submission for a
                                                                        ## context will run on a hardware engine before being preempted out to
                                                                        ## run a pending submission for another context.
        ("yieldTimeout", c_ulonglong)                                   ## [in,out] The maximum time in microseconds that the scheduler will wait
                                                                        ## to preempt a workload running on an engine before deciding to reset
                                                                        ## the hardware engine and terminating the associated context.
    ]

###############################################################################
## @brief Standby hardware components
class zes_standby_type_v(IntEnum):
    GLOBAL = 0                                      ## Control the overall standby policy of the device/sub-device

class zes_standby_type_t(c_int):
    def __str__(self):
        return str(zes_standby_type_v(self.value))


###############################################################################
## @brief Standby hardware component properties
class zes_standby_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("type", zes_standby_type_t),                                   ## [out] Which standby hardware component this controls
        ("onSubdevice", ze_bool_t),                                     ## [out] True if the resource is located on a sub-device; false means
                                                                        ## that the resource is on the device of the calling Sysman handle
        ("subdeviceId", c_ulong)                                        ## [out] If onSubdevice is true, this gives the ID of the sub-device
    ]

###############################################################################
## @brief Standby promotion modes
class zes_standby_promo_mode_v(IntEnum):
    DEFAULT = 0                                     ## Best compromise between performance and energy savings.
    NEVER = 1                                       ## The device/component will never shutdown. This can improve performance
                                                    ## but uses more energy.

class zes_standby_promo_mode_t(c_int):
    def __str__(self):
        return str(zes_standby_promo_mode_v(self.value))


###############################################################################
## @brief Temperature sensors
class zes_temp_sensors_v(IntEnum):
    GLOBAL = 0                                      ## The maximum temperature across all device sensors
    GPU = 1                                         ## The maximum temperature across all sensors in the GPU
    MEMORY = 2                                      ## The maximum temperature across all sensors in the local memory
    GLOBAL_MIN = 3                                  ## The minimum temperature across all device sensors
    GPU_MIN = 4                                     ## The minimum temperature across all sensors in the GPU
    MEMORY_MIN = 5                                  ## The minimum temperature across all sensors in the local device memory

class zes_temp_sensors_t(c_int):
    def __str__(self):
        return str(zes_temp_sensors_v(self.value))


###############################################################################
## @brief Temperature sensor properties
class zes_temp_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] pointer to extension-specific structure
        ("type", zes_temp_sensors_t),                                   ## [out] Which part of the device the temperature sensor measures
        ("onSubdevice", ze_bool_t),                                     ## [out] True if the resource is located on a sub-device; false means
                                                                        ## that the resource is on the device of the calling Sysman handle
        ("subdeviceId", c_ulong),                                       ## [out] If onSubdevice is true, this gives the ID of the sub-device
        ("maxTemperature", c_double),                                   ## [out] Will contain the maximum temperature for the specific device in
                                                                        ## degrees Celsius.
        ("isCriticalTempSupported", ze_bool_t),                         ## [out] Indicates if the critical temperature event
                                                                        ## ::ZES_EVENT_TYPE_FLAG_TEMP_CRITICAL is supported
        ("isThreshold1Supported", ze_bool_t),                           ## [out] Indicates if the temperature threshold 1 event
                                                                        ## ::ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD1 is supported
        ("isThreshold2Supported", ze_bool_t)                            ## [out] Indicates if the temperature threshold 2 event
                                                                        ## ::ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD2 is supported
    ]

###############################################################################
## @brief Temperature sensor threshold
class zes_temp_threshold_t(Structure):
    _fields_ = [
        ("enableLowToHigh", ze_bool_t),                                 ## [in,out] Trigger an event when the temperature crosses from below the
                                                                        ## threshold to above.
        ("enableHighToLow", ze_bool_t),                                 ## [in,out] Trigger an event when the temperature crosses from above the
                                                                        ## threshold to below.
        ("threshold", c_double)                                         ## [in,out] The threshold in degrees Celsius.
    ]

###############################################################################
## @brief Temperature configuration - which events should be triggered and the
##        trigger conditions.
class zes_temp_config_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] pointer to extension-specific structure
        ("enableCritical", ze_bool_t),                                  ## [in,out] Indicates if event ::ZES_EVENT_TYPE_FLAG_TEMP_CRITICAL should
                                                                        ## be triggered by the driver.
        ("threshold1", zes_temp_threshold_t),                           ## [in,out] Configuration controlling if and when event
                                                                        ## ::ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD1 should be triggered by the
                                                                        ## driver.
        ("threshold2", zes_temp_threshold_t)                            ## [in,out] Configuration controlling if and when event
                                                                        ## ::ZES_EVENT_TYPE_FLAG_TEMP_THRESHOLD2 should be triggered by the
                                                                        ## driver.
    ]

###############################################################################
__use_win_types = "Windows" == platform.uname()[0]

###############################################################################
## @brief Function-pointer for zesDriverEventListen
if __use_win_types:
    _zesDriverEventListen_t = WINFUNCTYPE( ze_result_t, ze_driver_handle_t, c_ulong, c_ulong, POINTER(zes_device_handle_t), POINTER(c_ulong), POINTER(zes_event_type_flags_t) )
else:
    _zesDriverEventListen_t = CFUNCTYPE( ze_result_t, ze_driver_handle_t, c_ulong, c_ulong, POINTER(zes_device_handle_t), POINTER(c_ulong), POINTER(zes_event_type_flags_t) )

###############################################################################
## @brief Function-pointer for zesDriverEventListenEx
if __use_win_types:
    _zesDriverEventListenEx_t = WINFUNCTYPE( ze_result_t, ze_driver_handle_t, c_ulonglong, c_ulong, POINTER(zes_device_handle_t), POINTER(c_ulong), POINTER(zes_event_type_flags_t) )
else:
    _zesDriverEventListenEx_t = CFUNCTYPE( ze_result_t, ze_driver_handle_t, c_ulonglong, c_ulong, POINTER(zes_device_handle_t), POINTER(c_ulong), POINTER(zes_event_type_flags_t) )


###############################################################################
## @brief Table of Driver functions pointers
class _zes_driver_dditable_t(Structure):
    _fields_ = [
        ("pfnEventListen", c_void_p),                                   ## _zesDriverEventListen_t
        ("pfnEventListenEx", c_void_p)                                  ## _zesDriverEventListenEx_t
    ]

###############################################################################
## @brief Function-pointer for zesDeviceGetProperties
if __use_win_types:
    _zesDeviceGetProperties_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(zes_device_properties_t) )
else:
    _zesDeviceGetProperties_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(zes_device_properties_t) )

###############################################################################
## @brief Function-pointer for zesDeviceGetState
if __use_win_types:
    _zesDeviceGetState_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(zes_device_state_t) )
else:
    _zesDeviceGetState_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(zes_device_state_t) )

###############################################################################
## @brief Function-pointer for zesDeviceReset
if __use_win_types:
    _zesDeviceReset_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, ze_bool_t )
else:
    _zesDeviceReset_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, ze_bool_t )

###############################################################################
## @brief Function-pointer for zesDeviceProcessesGetState
if __use_win_types:
    _zesDeviceProcessesGetState_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_process_state_t) )
else:
    _zesDeviceProcessesGetState_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_process_state_t) )

###############################################################################
## @brief Function-pointer for zesDevicePciGetProperties
if __use_win_types:
    _zesDevicePciGetProperties_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(zes_pci_properties_t) )
else:
    _zesDevicePciGetProperties_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(zes_pci_properties_t) )

###############################################################################
## @brief Function-pointer for zesDevicePciGetState
if __use_win_types:
    _zesDevicePciGetState_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(zes_pci_state_t) )
else:
    _zesDevicePciGetState_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(zes_pci_state_t) )

###############################################################################
## @brief Function-pointer for zesDevicePciGetBars
if __use_win_types:
    _zesDevicePciGetBars_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_pci_bar_properties_t) )
else:
    _zesDevicePciGetBars_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_pci_bar_properties_t) )

###############################################################################
## @brief Function-pointer for zesDevicePciGetStats
if __use_win_types:
    _zesDevicePciGetStats_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(zes_pci_stats_t) )
else:
    _zesDevicePciGetStats_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(zes_pci_stats_t) )

###############################################################################
## @brief Function-pointer for zesDeviceEnumDiagnosticTestSuites
if __use_win_types:
    _zesDeviceEnumDiagnosticTestSuites_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_diag_handle_t) )
else:
    _zesDeviceEnumDiagnosticTestSuites_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_diag_handle_t) )

###############################################################################
## @brief Function-pointer for zesDeviceEnumEngineGroups
if __use_win_types:
    _zesDeviceEnumEngineGroups_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_engine_handle_t) )
else:
    _zesDeviceEnumEngineGroups_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_engine_handle_t) )

###############################################################################
## @brief Function-pointer for zesDeviceEventRegister
if __use_win_types:
    _zesDeviceEventRegister_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, zes_event_type_flags_t )
else:
    _zesDeviceEventRegister_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, zes_event_type_flags_t )

###############################################################################
## @brief Function-pointer for zesDeviceEnumFabricPorts
if __use_win_types:
    _zesDeviceEnumFabricPorts_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_fabric_port_handle_t) )
else:
    _zesDeviceEnumFabricPorts_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_fabric_port_handle_t) )

###############################################################################
## @brief Function-pointer for zesDeviceEnumFans
if __use_win_types:
    _zesDeviceEnumFans_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_fan_handle_t) )
else:
    _zesDeviceEnumFans_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_fan_handle_t) )

###############################################################################
## @brief Function-pointer for zesDeviceEnumFirmwares
if __use_win_types:
    _zesDeviceEnumFirmwares_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_firmware_handle_t) )
else:
    _zesDeviceEnumFirmwares_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_firmware_handle_t) )

###############################################################################
## @brief Function-pointer for zesDeviceEnumFrequencyDomains
if __use_win_types:
    _zesDeviceEnumFrequencyDomains_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_freq_handle_t) )
else:
    _zesDeviceEnumFrequencyDomains_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_freq_handle_t) )

###############################################################################
## @brief Function-pointer for zesDeviceEnumLeds
if __use_win_types:
    _zesDeviceEnumLeds_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_led_handle_t) )
else:
    _zesDeviceEnumLeds_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_led_handle_t) )

###############################################################################
## @brief Function-pointer for zesDeviceEnumMemoryModules
if __use_win_types:
    _zesDeviceEnumMemoryModules_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_mem_handle_t) )
else:
    _zesDeviceEnumMemoryModules_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_mem_handle_t) )

###############################################################################
## @brief Function-pointer for zesDeviceEnumPerformanceFactorDomains
if __use_win_types:
    _zesDeviceEnumPerformanceFactorDomains_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_perf_handle_t) )
else:
    _zesDeviceEnumPerformanceFactorDomains_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_perf_handle_t) )

###############################################################################
## @brief Function-pointer for zesDeviceEnumPowerDomains
if __use_win_types:
    _zesDeviceEnumPowerDomains_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_pwr_handle_t) )
else:
    _zesDeviceEnumPowerDomains_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_pwr_handle_t) )

###############################################################################
## @brief Function-pointer for zesDeviceGetCardPowerDomain
if __use_win_types:
    _zesDeviceGetCardPowerDomain_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(zes_pwr_handle_t) )
else:
    _zesDeviceGetCardPowerDomain_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(zes_pwr_handle_t) )

###############################################################################
## @brief Function-pointer for zesDeviceEnumPsus
if __use_win_types:
    _zesDeviceEnumPsus_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_psu_handle_t) )
else:
    _zesDeviceEnumPsus_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_psu_handle_t) )

###############################################################################
## @brief Function-pointer for zesDeviceEnumRasErrorSets
if __use_win_types:
    _zesDeviceEnumRasErrorSets_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_ras_handle_t) )
else:
    _zesDeviceEnumRasErrorSets_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_ras_handle_t) )

###############################################################################
## @brief Function-pointer for zesDeviceEnumSchedulers
if __use_win_types:
    _zesDeviceEnumSchedulers_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_sched_handle_t) )
else:
    _zesDeviceEnumSchedulers_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_sched_handle_t) )

###############################################################################
## @brief Function-pointer for zesDeviceEnumStandbyDomains
if __use_win_types:
    _zesDeviceEnumStandbyDomains_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_standby_handle_t) )
else:
    _zesDeviceEnumStandbyDomains_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_standby_handle_t) )

###############################################################################
## @brief Function-pointer for zesDeviceEnumTemperatureSensors
if __use_win_types:
    _zesDeviceEnumTemperatureSensors_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_temp_handle_t) )
else:
    _zesDeviceEnumTemperatureSensors_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_temp_handle_t) )


###############################################################################
## @brief Table of Device functions pointers
class _zes_device_dditable_t(Structure):
    _fields_ = [
        ("pfnGetProperties", c_void_p),                                 ## _zesDeviceGetProperties_t
        ("pfnGetState", c_void_p),                                      ## _zesDeviceGetState_t
        ("pfnReset", c_void_p),                                         ## _zesDeviceReset_t
        ("pfnProcessesGetState", c_void_p),                             ## _zesDeviceProcessesGetState_t
        ("pfnPciGetProperties", c_void_p),                              ## _zesDevicePciGetProperties_t
        ("pfnPciGetState", c_void_p),                                   ## _zesDevicePciGetState_t
        ("pfnPciGetBars", c_void_p),                                    ## _zesDevicePciGetBars_t
        ("pfnPciGetStats", c_void_p),                                   ## _zesDevicePciGetStats_t
        ("pfnEnumDiagnosticTestSuites", c_void_p),                      ## _zesDeviceEnumDiagnosticTestSuites_t
        ("pfnEnumEngineGroups", c_void_p),                              ## _zesDeviceEnumEngineGroups_t
        ("pfnEventRegister", c_void_p),                                 ## _zesDeviceEventRegister_t
        ("pfnEnumFabricPorts", c_void_p),                               ## _zesDeviceEnumFabricPorts_t
        ("pfnEnumFans", c_void_p),                                      ## _zesDeviceEnumFans_t
        ("pfnEnumFirmwares", c_void_p),                                 ## _zesDeviceEnumFirmwares_t
        ("pfnEnumFrequencyDomains", c_void_p),                          ## _zesDeviceEnumFrequencyDomains_t
        ("pfnEnumLeds", c_void_p),                                      ## _zesDeviceEnumLeds_t
        ("pfnEnumMemoryModules", c_void_p),                             ## _zesDeviceEnumMemoryModules_t
        ("pfnEnumPerformanceFactorDomains", c_void_p),                  ## _zesDeviceEnumPerformanceFactorDomains_t
        ("pfnEnumPowerDomains", c_void_p),                              ## _zesDeviceEnumPowerDomains_t
        ("pfnGetCardPowerDomain", c_void_p),                            ## _zesDeviceGetCardPowerDomain_t
        ("pfnEnumPsus", c_void_p),                                      ## _zesDeviceEnumPsus_t
        ("pfnEnumRasErrorSets", c_void_p),                              ## _zesDeviceEnumRasErrorSets_t
        ("pfnEnumSchedulers", c_void_p),                                ## _zesDeviceEnumSchedulers_t
        ("pfnEnumStandbyDomains", c_void_p),                            ## _zesDeviceEnumStandbyDomains_t
        ("pfnEnumTemperatureSensors", c_void_p)                         ## _zesDeviceEnumTemperatureSensors_t
    ]

###############################################################################
## @brief Function-pointer for zesSchedulerGetProperties
if __use_win_types:
    _zesSchedulerGetProperties_t = WINFUNCTYPE( ze_result_t, zes_sched_handle_t, POINTER(zes_sched_properties_t) )
else:
    _zesSchedulerGetProperties_t = CFUNCTYPE( ze_result_t, zes_sched_handle_t, POINTER(zes_sched_properties_t) )

###############################################################################
## @brief Function-pointer for zesSchedulerGetCurrentMode
if __use_win_types:
    _zesSchedulerGetCurrentMode_t = WINFUNCTYPE( ze_result_t, zes_sched_handle_t, POINTER(zes_sched_mode_t) )
else:
    _zesSchedulerGetCurrentMode_t = CFUNCTYPE( ze_result_t, zes_sched_handle_t, POINTER(zes_sched_mode_t) )

###############################################################################
## @brief Function-pointer for zesSchedulerGetTimeoutModeProperties
if __use_win_types:
    _zesSchedulerGetTimeoutModeProperties_t = WINFUNCTYPE( ze_result_t, zes_sched_handle_t, ze_bool_t, POINTER(zes_sched_timeout_properties_t) )
else:
    _zesSchedulerGetTimeoutModeProperties_t = CFUNCTYPE( ze_result_t, zes_sched_handle_t, ze_bool_t, POINTER(zes_sched_timeout_properties_t) )

###############################################################################
## @brief Function-pointer for zesSchedulerGetTimesliceModeProperties
if __use_win_types:
    _zesSchedulerGetTimesliceModeProperties_t = WINFUNCTYPE( ze_result_t, zes_sched_handle_t, ze_bool_t, POINTER(zes_sched_timeslice_properties_t) )
else:
    _zesSchedulerGetTimesliceModeProperties_t = CFUNCTYPE( ze_result_t, zes_sched_handle_t, ze_bool_t, POINTER(zes_sched_timeslice_properties_t) )

###############################################################################
## @brief Function-pointer for zesSchedulerSetTimeoutMode
if __use_win_types:
    _zesSchedulerSetTimeoutMode_t = WINFUNCTYPE( ze_result_t, zes_sched_handle_t, POINTER(zes_sched_timeout_properties_t), POINTER(ze_bool_t) )
else:
    _zesSchedulerSetTimeoutMode_t = CFUNCTYPE( ze_result_t, zes_sched_handle_t, POINTER(zes_sched_timeout_properties_t), POINTER(ze_bool_t) )

###############################################################################
## @brief Function-pointer for zesSchedulerSetTimesliceMode
if __use_win_types:
    _zesSchedulerSetTimesliceMode_t = WINFUNCTYPE( ze_result_t, zes_sched_handle_t, POINTER(zes_sched_timeslice_properties_t), POINTER(ze_bool_t) )
else:
    _zesSchedulerSetTimesliceMode_t = CFUNCTYPE( ze_result_t, zes_sched_handle_t, POINTER(zes_sched_timeslice_properties_t), POINTER(ze_bool_t) )

###############################################################################
## @brief Function-pointer for zesSchedulerSetExclusiveMode
if __use_win_types:
    _zesSchedulerSetExclusiveMode_t = WINFUNCTYPE( ze_result_t, zes_sched_handle_t, POINTER(ze_bool_t) )
else:
    _zesSchedulerSetExclusiveMode_t = CFUNCTYPE( ze_result_t, zes_sched_handle_t, POINTER(ze_bool_t) )

###############################################################################
## @brief Function-pointer for zesSchedulerSetComputeUnitDebugMode
if __use_win_types:
    _zesSchedulerSetComputeUnitDebugMode_t = WINFUNCTYPE( ze_result_t, zes_sched_handle_t, POINTER(ze_bool_t) )
else:
    _zesSchedulerSetComputeUnitDebugMode_t = CFUNCTYPE( ze_result_t, zes_sched_handle_t, POINTER(ze_bool_t) )


###############################################################################
## @brief Table of Scheduler functions pointers
class _zes_scheduler_dditable_t(Structure):
    _fields_ = [
        ("pfnGetProperties", c_void_p),                                 ## _zesSchedulerGetProperties_t
        ("pfnGetCurrentMode", c_void_p),                                ## _zesSchedulerGetCurrentMode_t
        ("pfnGetTimeoutModeProperties", c_void_p),                      ## _zesSchedulerGetTimeoutModeProperties_t
        ("pfnGetTimesliceModeProperties", c_void_p),                    ## _zesSchedulerGetTimesliceModeProperties_t
        ("pfnSetTimeoutMode", c_void_p),                                ## _zesSchedulerSetTimeoutMode_t
        ("pfnSetTimesliceMode", c_void_p),                              ## _zesSchedulerSetTimesliceMode_t
        ("pfnSetExclusiveMode", c_void_p),                              ## _zesSchedulerSetExclusiveMode_t
        ("pfnSetComputeUnitDebugMode", c_void_p)                        ## _zesSchedulerSetComputeUnitDebugMode_t
    ]

###############################################################################
## @brief Function-pointer for zesPerformanceFactorGetProperties
if __use_win_types:
    _zesPerformanceFactorGetProperties_t = WINFUNCTYPE( ze_result_t, zes_perf_handle_t, POINTER(zes_perf_properties_t) )
else:
    _zesPerformanceFactorGetProperties_t = CFUNCTYPE( ze_result_t, zes_perf_handle_t, POINTER(zes_perf_properties_t) )

###############################################################################
## @brief Function-pointer for zesPerformanceFactorGetConfig
if __use_win_types:
    _zesPerformanceFactorGetConfig_t = WINFUNCTYPE( ze_result_t, zes_perf_handle_t, POINTER(c_double) )
else:
    _zesPerformanceFactorGetConfig_t = CFUNCTYPE( ze_result_t, zes_perf_handle_t, POINTER(c_double) )

###############################################################################
## @brief Function-pointer for zesPerformanceFactorSetConfig
if __use_win_types:
    _zesPerformanceFactorSetConfig_t = WINFUNCTYPE( ze_result_t, zes_perf_handle_t, c_double )
else:
    _zesPerformanceFactorSetConfig_t = CFUNCTYPE( ze_result_t, zes_perf_handle_t, c_double )


###############################################################################
## @brief Table of PerformanceFactor functions pointers
class _zes_performance_factor_dditable_t(Structure):
    _fields_ = [
        ("pfnGetProperties", c_void_p),                                 ## _zesPerformanceFactorGetProperties_t
        ("pfnGetConfig", c_void_p),                                     ## _zesPerformanceFactorGetConfig_t
        ("pfnSetConfig", c_void_p)                                      ## _zesPerformanceFactorSetConfig_t
    ]

###############################################################################
## @brief Function-pointer for zesPowerGetProperties
if __use_win_types:
    _zesPowerGetProperties_t = WINFUNCTYPE( ze_result_t, zes_pwr_handle_t, POINTER(zes_power_properties_t) )
else:
    _zesPowerGetProperties_t = CFUNCTYPE( ze_result_t, zes_pwr_handle_t, POINTER(zes_power_properties_t) )

###############################################################################
## @brief Function-pointer for zesPowerGetEnergyCounter
if __use_win_types:
    _zesPowerGetEnergyCounter_t = WINFUNCTYPE( ze_result_t, zes_pwr_handle_t, POINTER(zes_power_energy_counter_t) )
else:
    _zesPowerGetEnergyCounter_t = CFUNCTYPE( ze_result_t, zes_pwr_handle_t, POINTER(zes_power_energy_counter_t) )

###############################################################################
## @brief Function-pointer for zesPowerGetLimits
if __use_win_types:
    _zesPowerGetLimits_t = WINFUNCTYPE( ze_result_t, zes_pwr_handle_t, POINTER(zes_power_sustained_limit_t), POINTER(zes_power_burst_limit_t), POINTER(zes_power_peak_limit_t) )
else:
    _zesPowerGetLimits_t = CFUNCTYPE( ze_result_t, zes_pwr_handle_t, POINTER(zes_power_sustained_limit_t), POINTER(zes_power_burst_limit_t), POINTER(zes_power_peak_limit_t) )

###############################################################################
## @brief Function-pointer for zesPowerSetLimits
if __use_win_types:
    _zesPowerSetLimits_t = WINFUNCTYPE( ze_result_t, zes_pwr_handle_t, POINTER(zes_power_sustained_limit_t), POINTER(zes_power_burst_limit_t), POINTER(zes_power_peak_limit_t) )
else:
    _zesPowerSetLimits_t = CFUNCTYPE( ze_result_t, zes_pwr_handle_t, POINTER(zes_power_sustained_limit_t), POINTER(zes_power_burst_limit_t), POINTER(zes_power_peak_limit_t) )

###############################################################################
## @brief Function-pointer for zesPowerGetEnergyThreshold
if __use_win_types:
    _zesPowerGetEnergyThreshold_t = WINFUNCTYPE( ze_result_t, zes_pwr_handle_t, POINTER(zes_energy_threshold_t) )
else:
    _zesPowerGetEnergyThreshold_t = CFUNCTYPE( ze_result_t, zes_pwr_handle_t, POINTER(zes_energy_threshold_t) )

###############################################################################
## @brief Function-pointer for zesPowerSetEnergyThreshold
if __use_win_types:
    _zesPowerSetEnergyThreshold_t = WINFUNCTYPE( ze_result_t, zes_pwr_handle_t, c_double )
else:
    _zesPowerSetEnergyThreshold_t = CFUNCTYPE( ze_result_t, zes_pwr_handle_t, c_double )


###############################################################################
## @brief Table of Power functions pointers
class _zes_power_dditable_t(Structure):
    _fields_ = [
        ("pfnGetProperties", c_void_p),                                 ## _zesPowerGetProperties_t
        ("pfnGetEnergyCounter", c_void_p),                              ## _zesPowerGetEnergyCounter_t
        ("pfnGetLimits", c_void_p),                                     ## _zesPowerGetLimits_t
        ("pfnSetLimits", c_void_p),                                     ## _zesPowerSetLimits_t
        ("pfnGetEnergyThreshold", c_void_p),                            ## _zesPowerGetEnergyThreshold_t
        ("pfnSetEnergyThreshold", c_void_p)                             ## _zesPowerSetEnergyThreshold_t
    ]

###############################################################################
## @brief Function-pointer for zesFrequencyGetProperties
if __use_win_types:
    _zesFrequencyGetProperties_t = WINFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(zes_freq_properties_t) )
else:
    _zesFrequencyGetProperties_t = CFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(zes_freq_properties_t) )

###############################################################################
## @brief Function-pointer for zesFrequencyGetAvailableClocks
if __use_win_types:
    _zesFrequencyGetAvailableClocks_t = WINFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(c_ulong), POINTER(c_double) )
else:
    _zesFrequencyGetAvailableClocks_t = CFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(c_ulong), POINTER(c_double) )

###############################################################################
## @brief Function-pointer for zesFrequencyGetRange
if __use_win_types:
    _zesFrequencyGetRange_t = WINFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(zes_freq_range_t) )
else:
    _zesFrequencyGetRange_t = CFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(zes_freq_range_t) )

###############################################################################
## @brief Function-pointer for zesFrequencySetRange
if __use_win_types:
    _zesFrequencySetRange_t = WINFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(zes_freq_range_t) )
else:
    _zesFrequencySetRange_t = CFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(zes_freq_range_t) )

###############################################################################
## @brief Function-pointer for zesFrequencyGetState
if __use_win_types:
    _zesFrequencyGetState_t = WINFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(zes_freq_state_t) )
else:
    _zesFrequencyGetState_t = CFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(zes_freq_state_t) )

###############################################################################
## @brief Function-pointer for zesFrequencyGetThrottleTime
if __use_win_types:
    _zesFrequencyGetThrottleTime_t = WINFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(zes_freq_throttle_time_t) )
else:
    _zesFrequencyGetThrottleTime_t = CFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(zes_freq_throttle_time_t) )

###############################################################################
## @brief Function-pointer for zesFrequencyOcGetCapabilities
if __use_win_types:
    _zesFrequencyOcGetCapabilities_t = WINFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(zes_oc_capabilities_t) )
else:
    _zesFrequencyOcGetCapabilities_t = CFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(zes_oc_capabilities_t) )

###############################################################################
## @brief Function-pointer for zesFrequencyOcGetFrequencyTarget
if __use_win_types:
    _zesFrequencyOcGetFrequencyTarget_t = WINFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(c_double) )
else:
    _zesFrequencyOcGetFrequencyTarget_t = CFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(c_double) )

###############################################################################
## @brief Function-pointer for zesFrequencyOcSetFrequencyTarget
if __use_win_types:
    _zesFrequencyOcSetFrequencyTarget_t = WINFUNCTYPE( ze_result_t, zes_freq_handle_t, c_double )
else:
    _zesFrequencyOcSetFrequencyTarget_t = CFUNCTYPE( ze_result_t, zes_freq_handle_t, c_double )

###############################################################################
## @brief Function-pointer for zesFrequencyOcGetVoltageTarget
if __use_win_types:
    _zesFrequencyOcGetVoltageTarget_t = WINFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(c_double), POINTER(c_double) )
else:
    _zesFrequencyOcGetVoltageTarget_t = CFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(c_double), POINTER(c_double) )

###############################################################################
## @brief Function-pointer for zesFrequencyOcSetVoltageTarget
if __use_win_types:
    _zesFrequencyOcSetVoltageTarget_t = WINFUNCTYPE( ze_result_t, zes_freq_handle_t, c_double, c_double )
else:
    _zesFrequencyOcSetVoltageTarget_t = CFUNCTYPE( ze_result_t, zes_freq_handle_t, c_double, c_double )

###############################################################################
## @brief Function-pointer for zesFrequencyOcSetMode
if __use_win_types:
    _zesFrequencyOcSetMode_t = WINFUNCTYPE( ze_result_t, zes_freq_handle_t, zes_oc_mode_t )
else:
    _zesFrequencyOcSetMode_t = CFUNCTYPE( ze_result_t, zes_freq_handle_t, zes_oc_mode_t )

###############################################################################
## @brief Function-pointer for zesFrequencyOcGetMode
if __use_win_types:
    _zesFrequencyOcGetMode_t = WINFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(zes_oc_mode_t) )
else:
    _zesFrequencyOcGetMode_t = CFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(zes_oc_mode_t) )

###############################################################################
## @brief Function-pointer for zesFrequencyOcGetIccMax
if __use_win_types:
    _zesFrequencyOcGetIccMax_t = WINFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(c_double) )
else:
    _zesFrequencyOcGetIccMax_t = CFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(c_double) )

###############################################################################
## @brief Function-pointer for zesFrequencyOcSetIccMax
if __use_win_types:
    _zesFrequencyOcSetIccMax_t = WINFUNCTYPE( ze_result_t, zes_freq_handle_t, c_double )
else:
    _zesFrequencyOcSetIccMax_t = CFUNCTYPE( ze_result_t, zes_freq_handle_t, c_double )

###############################################################################
## @brief Function-pointer for zesFrequencyOcGetTjMax
if __use_win_types:
    _zesFrequencyOcGetTjMax_t = WINFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(c_double) )
else:
    _zesFrequencyOcGetTjMax_t = CFUNCTYPE( ze_result_t, zes_freq_handle_t, POINTER(c_double) )

###############################################################################
## @brief Function-pointer for zesFrequencyOcSetTjMax
if __use_win_types:
    _zesFrequencyOcSetTjMax_t = WINFUNCTYPE( ze_result_t, zes_freq_handle_t, c_double )
else:
    _zesFrequencyOcSetTjMax_t = CFUNCTYPE( ze_result_t, zes_freq_handle_t, c_double )


###############################################################################
## @brief Table of Frequency functions pointers
class _zes_frequency_dditable_t(Structure):
    _fields_ = [
        ("pfnGetProperties", c_void_p),                                 ## _zesFrequencyGetProperties_t
        ("pfnGetAvailableClocks", c_void_p),                            ## _zesFrequencyGetAvailableClocks_t
        ("pfnGetRange", c_void_p),                                      ## _zesFrequencyGetRange_t
        ("pfnSetRange", c_void_p),                                      ## _zesFrequencySetRange_t
        ("pfnGetState", c_void_p),                                      ## _zesFrequencyGetState_t
        ("pfnGetThrottleTime", c_void_p),                               ## _zesFrequencyGetThrottleTime_t
        ("pfnOcGetCapabilities", c_void_p),                             ## _zesFrequencyOcGetCapabilities_t
        ("pfnOcGetFrequencyTarget", c_void_p),                          ## _zesFrequencyOcGetFrequencyTarget_t
        ("pfnOcSetFrequencyTarget", c_void_p),                          ## _zesFrequencyOcSetFrequencyTarget_t
        ("pfnOcGetVoltageTarget", c_void_p),                            ## _zesFrequencyOcGetVoltageTarget_t
        ("pfnOcSetVoltageTarget", c_void_p),                            ## _zesFrequencyOcSetVoltageTarget_t
        ("pfnOcSetMode", c_void_p),                                     ## _zesFrequencyOcSetMode_t
        ("pfnOcGetMode", c_void_p),                                     ## _zesFrequencyOcGetMode_t
        ("pfnOcGetIccMax", c_void_p),                                   ## _zesFrequencyOcGetIccMax_t
        ("pfnOcSetIccMax", c_void_p),                                   ## _zesFrequencyOcSetIccMax_t
        ("pfnOcGetTjMax", c_void_p),                                    ## _zesFrequencyOcGetTjMax_t
        ("pfnOcSetTjMax", c_void_p)                                     ## _zesFrequencyOcSetTjMax_t
    ]

###############################################################################
## @brief Function-pointer for zesEngineGetProperties
if __use_win_types:
    _zesEngineGetProperties_t = WINFUNCTYPE( ze_result_t, zes_engine_handle_t, POINTER(zes_engine_properties_t) )
else:
    _zesEngineGetProperties_t = CFUNCTYPE( ze_result_t, zes_engine_handle_t, POINTER(zes_engine_properties_t) )

###############################################################################
## @brief Function-pointer for zesEngineGetActivity
if __use_win_types:
    _zesEngineGetActivity_t = WINFUNCTYPE( ze_result_t, zes_engine_handle_t, POINTER(zes_engine_stats_t) )
else:
    _zesEngineGetActivity_t = CFUNCTYPE( ze_result_t, zes_engine_handle_t, POINTER(zes_engine_stats_t) )


###############################################################################
## @brief Table of Engine functions pointers
class _zes_engine_dditable_t(Structure):
    _fields_ = [
        ("pfnGetProperties", c_void_p),                                 ## _zesEngineGetProperties_t
        ("pfnGetActivity", c_void_p)                                    ## _zesEngineGetActivity_t
    ]

###############################################################################
## @brief Function-pointer for zesStandbyGetProperties
if __use_win_types:
    _zesStandbyGetProperties_t = WINFUNCTYPE( ze_result_t, zes_standby_handle_t, POINTER(zes_standby_properties_t) )
else:
    _zesStandbyGetProperties_t = CFUNCTYPE( ze_result_t, zes_standby_handle_t, POINTER(zes_standby_properties_t) )

###############################################################################
## @brief Function-pointer for zesStandbyGetMode
if __use_win_types:
    _zesStandbyGetMode_t = WINFUNCTYPE( ze_result_t, zes_standby_handle_t, POINTER(zes_standby_promo_mode_t) )
else:
    _zesStandbyGetMode_t = CFUNCTYPE( ze_result_t, zes_standby_handle_t, POINTER(zes_standby_promo_mode_t) )

###############################################################################
## @brief Function-pointer for zesStandbySetMode
if __use_win_types:
    _zesStandbySetMode_t = WINFUNCTYPE( ze_result_t, zes_standby_handle_t, zes_standby_promo_mode_t )
else:
    _zesStandbySetMode_t = CFUNCTYPE( ze_result_t, zes_standby_handle_t, zes_standby_promo_mode_t )


###############################################################################
## @brief Table of Standby functions pointers
class _zes_standby_dditable_t(Structure):
    _fields_ = [
        ("pfnGetProperties", c_void_p),                                 ## _zesStandbyGetProperties_t
        ("pfnGetMode", c_void_p),                                       ## _zesStandbyGetMode_t
        ("pfnSetMode", c_void_p)                                        ## _zesStandbySetMode_t
    ]

###############################################################################
## @brief Function-pointer for zesFirmwareGetProperties
if __use_win_types:
    _zesFirmwareGetProperties_t = WINFUNCTYPE( ze_result_t, zes_firmware_handle_t, POINTER(zes_firmware_properties_t) )
else:
    _zesFirmwareGetProperties_t = CFUNCTYPE( ze_result_t, zes_firmware_handle_t, POINTER(zes_firmware_properties_t) )

###############################################################################
## @brief Function-pointer for zesFirmwareFlash
if __use_win_types:
    _zesFirmwareFlash_t = WINFUNCTYPE( ze_result_t, zes_firmware_handle_t, c_void_p, c_ulong )
else:
    _zesFirmwareFlash_t = CFUNCTYPE( ze_result_t, zes_firmware_handle_t, c_void_p, c_ulong )


###############################################################################
## @brief Table of Firmware functions pointers
class _zes_firmware_dditable_t(Structure):
    _fields_ = [
        ("pfnGetProperties", c_void_p),                                 ## _zesFirmwareGetProperties_t
        ("pfnFlash", c_void_p)                                          ## _zesFirmwareFlash_t
    ]

###############################################################################
## @brief Function-pointer for zesMemoryGetProperties
if __use_win_types:
    _zesMemoryGetProperties_t = WINFUNCTYPE( ze_result_t, zes_mem_handle_t, POINTER(zes_mem_properties_t) )
else:
    _zesMemoryGetProperties_t = CFUNCTYPE( ze_result_t, zes_mem_handle_t, POINTER(zes_mem_properties_t) )

###############################################################################
## @brief Function-pointer for zesMemoryGetState
if __use_win_types:
    _zesMemoryGetState_t = WINFUNCTYPE( ze_result_t, zes_mem_handle_t, POINTER(zes_mem_state_t) )
else:
    _zesMemoryGetState_t = CFUNCTYPE( ze_result_t, zes_mem_handle_t, POINTER(zes_mem_state_t) )

###############################################################################
## @brief Function-pointer for zesMemoryGetBandwidth
if __use_win_types:
    _zesMemoryGetBandwidth_t = WINFUNCTYPE( ze_result_t, zes_mem_handle_t, POINTER(zes_mem_bandwidth_t) )
else:
    _zesMemoryGetBandwidth_t = CFUNCTYPE( ze_result_t, zes_mem_handle_t, POINTER(zes_mem_bandwidth_t) )


###############################################################################
## @brief Table of Memory functions pointers
class _zes_memory_dditable_t(Structure):
    _fields_ = [
        ("pfnGetProperties", c_void_p),                                 ## _zesMemoryGetProperties_t
        ("pfnGetState", c_void_p),                                      ## _zesMemoryGetState_t
        ("pfnGetBandwidth", c_void_p)                                   ## _zesMemoryGetBandwidth_t
    ]

###############################################################################
## @brief Function-pointer for zesFabricPortGetProperties
if __use_win_types:
    _zesFabricPortGetProperties_t = WINFUNCTYPE( ze_result_t, zes_fabric_port_handle_t, POINTER(zes_fabric_port_properties_t) )
else:
    _zesFabricPortGetProperties_t = CFUNCTYPE( ze_result_t, zes_fabric_port_handle_t, POINTER(zes_fabric_port_properties_t) )

###############################################################################
## @brief Function-pointer for zesFabricPortGetLinkType
if __use_win_types:
    _zesFabricPortGetLinkType_t = WINFUNCTYPE( ze_result_t, zes_fabric_port_handle_t, POINTER(zes_fabric_link_type_t) )
else:
    _zesFabricPortGetLinkType_t = CFUNCTYPE( ze_result_t, zes_fabric_port_handle_t, POINTER(zes_fabric_link_type_t) )

###############################################################################
## @brief Function-pointer for zesFabricPortGetConfig
if __use_win_types:
    _zesFabricPortGetConfig_t = WINFUNCTYPE( ze_result_t, zes_fabric_port_handle_t, POINTER(zes_fabric_port_config_t) )
else:
    _zesFabricPortGetConfig_t = CFUNCTYPE( ze_result_t, zes_fabric_port_handle_t, POINTER(zes_fabric_port_config_t) )

###############################################################################
## @brief Function-pointer for zesFabricPortSetConfig
if __use_win_types:
    _zesFabricPortSetConfig_t = WINFUNCTYPE( ze_result_t, zes_fabric_port_handle_t, POINTER(zes_fabric_port_config_t) )
else:
    _zesFabricPortSetConfig_t = CFUNCTYPE( ze_result_t, zes_fabric_port_handle_t, POINTER(zes_fabric_port_config_t) )

###############################################################################
## @brief Function-pointer for zesFabricPortGetState
if __use_win_types:
    _zesFabricPortGetState_t = WINFUNCTYPE( ze_result_t, zes_fabric_port_handle_t, POINTER(zes_fabric_port_state_t) )
else:
    _zesFabricPortGetState_t = CFUNCTYPE( ze_result_t, zes_fabric_port_handle_t, POINTER(zes_fabric_port_state_t) )

###############################################################################
## @brief Function-pointer for zesFabricPortGetThroughput
if __use_win_types:
    _zesFabricPortGetThroughput_t = WINFUNCTYPE( ze_result_t, zes_fabric_port_handle_t, POINTER(zes_fabric_port_throughput_t) )
else:
    _zesFabricPortGetThroughput_t = CFUNCTYPE( ze_result_t, zes_fabric_port_handle_t, POINTER(zes_fabric_port_throughput_t) )


###############################################################################
## @brief Table of FabricPort functions pointers
class _zes_fabric_port_dditable_t(Structure):
    _fields_ = [
        ("pfnGetProperties", c_void_p),                                 ## _zesFabricPortGetProperties_t
        ("pfnGetLinkType", c_void_p),                                   ## _zesFabricPortGetLinkType_t
        ("pfnGetConfig", c_void_p),                                     ## _zesFabricPortGetConfig_t
        ("pfnSetConfig", c_void_p),                                     ## _zesFabricPortSetConfig_t
        ("pfnGetState", c_void_p),                                      ## _zesFabricPortGetState_t
        ("pfnGetThroughput", c_void_p)                                  ## _zesFabricPortGetThroughput_t
    ]

###############################################################################
## @brief Function-pointer for zesTemperatureGetProperties
if __use_win_types:
    _zesTemperatureGetProperties_t = WINFUNCTYPE( ze_result_t, zes_temp_handle_t, POINTER(zes_temp_properties_t) )
else:
    _zesTemperatureGetProperties_t = CFUNCTYPE( ze_result_t, zes_temp_handle_t, POINTER(zes_temp_properties_t) )

###############################################################################
## @brief Function-pointer for zesTemperatureGetConfig
if __use_win_types:
    _zesTemperatureGetConfig_t = WINFUNCTYPE( ze_result_t, zes_temp_handle_t, POINTER(zes_temp_config_t) )
else:
    _zesTemperatureGetConfig_t = CFUNCTYPE( ze_result_t, zes_temp_handle_t, POINTER(zes_temp_config_t) )

###############################################################################
## @brief Function-pointer for zesTemperatureSetConfig
if __use_win_types:
    _zesTemperatureSetConfig_t = WINFUNCTYPE( ze_result_t, zes_temp_handle_t, POINTER(zes_temp_config_t) )
else:
    _zesTemperatureSetConfig_t = CFUNCTYPE( ze_result_t, zes_temp_handle_t, POINTER(zes_temp_config_t) )

###############################################################################
## @brief Function-pointer for zesTemperatureGetState
if __use_win_types:
    _zesTemperatureGetState_t = WINFUNCTYPE( ze_result_t, zes_temp_handle_t, POINTER(c_double) )
else:
    _zesTemperatureGetState_t = CFUNCTYPE( ze_result_t, zes_temp_handle_t, POINTER(c_double) )


###############################################################################
## @brief Table of Temperature functions pointers
class _zes_temperature_dditable_t(Structure):
    _fields_ = [
        ("pfnGetProperties", c_void_p),                                 ## _zesTemperatureGetProperties_t
        ("pfnGetConfig", c_void_p),                                     ## _zesTemperatureGetConfig_t
        ("pfnSetConfig", c_void_p),                                     ## _zesTemperatureSetConfig_t
        ("pfnGetState", c_void_p)                                       ## _zesTemperatureGetState_t
    ]

###############################################################################
## @brief Function-pointer for zesPsuGetProperties
if __use_win_types:
    _zesPsuGetProperties_t = WINFUNCTYPE( ze_result_t, zes_psu_handle_t, POINTER(zes_psu_properties_t) )
else:
    _zesPsuGetProperties_t = CFUNCTYPE( ze_result_t, zes_psu_handle_t, POINTER(zes_psu_properties_t) )

###############################################################################
## @brief Function-pointer for zesPsuGetState
if __use_win_types:
    _zesPsuGetState_t = WINFUNCTYPE( ze_result_t, zes_psu_handle_t, POINTER(zes_psu_state_t) )
else:
    _zesPsuGetState_t = CFUNCTYPE( ze_result_t, zes_psu_handle_t, POINTER(zes_psu_state_t) )


###############################################################################
## @brief Table of Psu functions pointers
class _zes_psu_dditable_t(Structure):
    _fields_ = [
        ("pfnGetProperties", c_void_p),                                 ## _zesPsuGetProperties_t
        ("pfnGetState", c_void_p)                                       ## _zesPsuGetState_t
    ]

###############################################################################
## @brief Function-pointer for zesFanGetProperties
if __use_win_types:
    _zesFanGetProperties_t = WINFUNCTYPE( ze_result_t, zes_fan_handle_t, POINTER(zes_fan_properties_t) )
else:
    _zesFanGetProperties_t = CFUNCTYPE( ze_result_t, zes_fan_handle_t, POINTER(zes_fan_properties_t) )

###############################################################################
## @brief Function-pointer for zesFanGetConfig
if __use_win_types:
    _zesFanGetConfig_t = WINFUNCTYPE( ze_result_t, zes_fan_handle_t, POINTER(zes_fan_config_t) )
else:
    _zesFanGetConfig_t = CFUNCTYPE( ze_result_t, zes_fan_handle_t, POINTER(zes_fan_config_t) )

###############################################################################
## @brief Function-pointer for zesFanSetDefaultMode
if __use_win_types:
    _zesFanSetDefaultMode_t = WINFUNCTYPE( ze_result_t, zes_fan_handle_t )
else:
    _zesFanSetDefaultMode_t = CFUNCTYPE( ze_result_t, zes_fan_handle_t )

###############################################################################
## @brief Function-pointer for zesFanSetFixedSpeedMode
if __use_win_types:
    _zesFanSetFixedSpeedMode_t = WINFUNCTYPE( ze_result_t, zes_fan_handle_t, POINTER(zes_fan_speed_t) )
else:
    _zesFanSetFixedSpeedMode_t = CFUNCTYPE( ze_result_t, zes_fan_handle_t, POINTER(zes_fan_speed_t) )

###############################################################################
## @brief Function-pointer for zesFanSetSpeedTableMode
if __use_win_types:
    _zesFanSetSpeedTableMode_t = WINFUNCTYPE( ze_result_t, zes_fan_handle_t, POINTER(zes_fan_speed_table_t) )
else:
    _zesFanSetSpeedTableMode_t = CFUNCTYPE( ze_result_t, zes_fan_handle_t, POINTER(zes_fan_speed_table_t) )

###############################################################################
## @brief Function-pointer for zesFanGetState
if __use_win_types:
    _zesFanGetState_t = WINFUNCTYPE( ze_result_t, zes_fan_handle_t, zes_fan_speed_units_t, POINTER(c_int32_t) )
else:
    _zesFanGetState_t = CFUNCTYPE( ze_result_t, zes_fan_handle_t, zes_fan_speed_units_t, POINTER(c_int32_t) )


###############################################################################
## @brief Table of Fan functions pointers
class _zes_fan_dditable_t(Structure):
    _fields_ = [
        ("pfnGetProperties", c_void_p),                                 ## _zesFanGetProperties_t
        ("pfnGetConfig", c_void_p),                                     ## _zesFanGetConfig_t
        ("pfnSetDefaultMode", c_void_p),                                ## _zesFanSetDefaultMode_t
        ("pfnSetFixedSpeedMode", c_void_p),                             ## _zesFanSetFixedSpeedMode_t
        ("pfnSetSpeedTableMode", c_void_p),                             ## _zesFanSetSpeedTableMode_t
        ("pfnGetState", c_void_p)                                       ## _zesFanGetState_t
    ]

###############################################################################
## @brief Function-pointer for zesLedGetProperties
if __use_win_types:
    _zesLedGetProperties_t = WINFUNCTYPE( ze_result_t, zes_led_handle_t, POINTER(zes_led_properties_t) )
else:
    _zesLedGetProperties_t = CFUNCTYPE( ze_result_t, zes_led_handle_t, POINTER(zes_led_properties_t) )

###############################################################################
## @brief Function-pointer for zesLedGetState
if __use_win_types:
    _zesLedGetState_t = WINFUNCTYPE( ze_result_t, zes_led_handle_t, POINTER(zes_led_state_t) )
else:
    _zesLedGetState_t = CFUNCTYPE( ze_result_t, zes_led_handle_t, POINTER(zes_led_state_t) )

###############################################################################
## @brief Function-pointer for zesLedSetState
if __use_win_types:
    _zesLedSetState_t = WINFUNCTYPE( ze_result_t, zes_led_handle_t, ze_bool_t )
else:
    _zesLedSetState_t = CFUNCTYPE( ze_result_t, zes_led_handle_t, ze_bool_t )

###############################################################################
## @brief Function-pointer for zesLedSetColor
if __use_win_types:
    _zesLedSetColor_t = WINFUNCTYPE( ze_result_t, zes_led_handle_t, POINTER(zes_led_color_t) )
else:
    _zesLedSetColor_t = CFUNCTYPE( ze_result_t, zes_led_handle_t, POINTER(zes_led_color_t) )


###############################################################################
## @brief Table of Led functions pointers
class _zes_led_dditable_t(Structure):
    _fields_ = [
        ("pfnGetProperties", c_void_p),                                 ## _zesLedGetProperties_t
        ("pfnGetState", c_void_p),                                      ## _zesLedGetState_t
        ("pfnSetState", c_void_p),                                      ## _zesLedSetState_t
        ("pfnSetColor", c_void_p)                                       ## _zesLedSetColor_t
    ]

###############################################################################
## @brief Function-pointer for zesRasGetProperties
if __use_win_types:
    _zesRasGetProperties_t = WINFUNCTYPE( ze_result_t, zes_ras_handle_t, POINTER(zes_ras_properties_t) )
else:
    _zesRasGetProperties_t = CFUNCTYPE( ze_result_t, zes_ras_handle_t, POINTER(zes_ras_properties_t) )

###############################################################################
## @brief Function-pointer for zesRasGetConfig
if __use_win_types:
    _zesRasGetConfig_t = WINFUNCTYPE( ze_result_t, zes_ras_handle_t, POINTER(zes_ras_config_t) )
else:
    _zesRasGetConfig_t = CFUNCTYPE( ze_result_t, zes_ras_handle_t, POINTER(zes_ras_config_t) )

###############################################################################
## @brief Function-pointer for zesRasSetConfig
if __use_win_types:
    _zesRasSetConfig_t = WINFUNCTYPE( ze_result_t, zes_ras_handle_t, POINTER(zes_ras_config_t) )
else:
    _zesRasSetConfig_t = CFUNCTYPE( ze_result_t, zes_ras_handle_t, POINTER(zes_ras_config_t) )

###############################################################################
## @brief Function-pointer for zesRasGetState
if __use_win_types:
    _zesRasGetState_t = WINFUNCTYPE( ze_result_t, zes_ras_handle_t, ze_bool_t, POINTER(zes_ras_state_t) )
else:
    _zesRasGetState_t = CFUNCTYPE( ze_result_t, zes_ras_handle_t, ze_bool_t, POINTER(zes_ras_state_t) )


###############################################################################
## @brief Table of Ras functions pointers
class _zes_ras_dditable_t(Structure):
    _fields_ = [
        ("pfnGetProperties", c_void_p),                                 ## _zesRasGetProperties_t
        ("pfnGetConfig", c_void_p),                                     ## _zesRasGetConfig_t
        ("pfnSetConfig", c_void_p),                                     ## _zesRasSetConfig_t
        ("pfnGetState", c_void_p)                                       ## _zesRasGetState_t
    ]

###############################################################################
## @brief Function-pointer for zesDiagnosticsGetProperties
if __use_win_types:
    _zesDiagnosticsGetProperties_t = WINFUNCTYPE( ze_result_t, zes_diag_handle_t, POINTER(zes_diag_properties_t) )
else:
    _zesDiagnosticsGetProperties_t = CFUNCTYPE( ze_result_t, zes_diag_handle_t, POINTER(zes_diag_properties_t) )

###############################################################################
## @brief Function-pointer for zesDiagnosticsGetTests
if __use_win_types:
    _zesDiagnosticsGetTests_t = WINFUNCTYPE( ze_result_t, zes_diag_handle_t, POINTER(c_ulong), POINTER(zes_diag_test_t) )
else:
    _zesDiagnosticsGetTests_t = CFUNCTYPE( ze_result_t, zes_diag_handle_t, POINTER(c_ulong), POINTER(zes_diag_test_t) )

###############################################################################
## @brief Function-pointer for zesDiagnosticsRunTests
if __use_win_types:
    _zesDiagnosticsRunTests_t = WINFUNCTYPE( ze_result_t, zes_diag_handle_t, c_ulong, c_ulong, POINTER(zes_diag_result_t) )
else:
    _zesDiagnosticsRunTests_t = CFUNCTYPE( ze_result_t, zes_diag_handle_t, c_ulong, c_ulong, POINTER(zes_diag_result_t) )


###############################################################################
## @brief Table of Diagnostics functions pointers
class _zes_diagnostics_dditable_t(Structure):
    _fields_ = [
        ("pfnGetProperties", c_void_p),                                 ## _zesDiagnosticsGetProperties_t
        ("pfnGetTests", c_void_p),                                      ## _zesDiagnosticsGetTests_t
        ("pfnRunTests", c_void_p)                                       ## _zesDiagnosticsRunTests_t
    ]

###############################################################################
class _zes_dditable_t(Structure):
    _fields_ = [
        ("Driver", _zes_driver_dditable_t),
        ("Device", _zes_device_dditable_t),
        ("Scheduler", _zes_scheduler_dditable_t),
        ("PerformanceFactor", _zes_performance_factor_dditable_t),
        ("Power", _zes_power_dditable_t),
        ("Frequency", _zes_frequency_dditable_t),
        ("Engine", _zes_engine_dditable_t),
        ("Standby", _zes_standby_dditable_t),
        ("Firmware", _zes_firmware_dditable_t),
        ("Memory", _zes_memory_dditable_t),
        ("FabricPort", _zes_fabric_port_dditable_t),
        ("Temperature", _zes_temperature_dditable_t),
        ("Psu", _zes_psu_dditable_t),
        ("Fan", _zes_fan_dditable_t),
        ("Led", _zes_led_dditable_t),
        ("Ras", _zes_ras_dditable_t),
        ("Diagnostics", _zes_diagnostics_dditable_t)
    ]

###############################################################################
## @brief zes device-driver interfaces
class ZES_DDI:
    def __init__(self, version : ze_api_version_t):
        # load the ze_loader library
        if "Windows" == platform.uname()[0]:
            self.__dll = WinDLL("ze_loader.dll")
        else:
            self.__dll = CDLL("ze_loader.so")

        # fill the ddi tables
        self.__dditable = _zes_dditable_t()

        # call driver to get function pointers
        _Driver = _zes_driver_dditable_t()
        r = ze_result_v(self.__dll.zesGetDriverProcAddrTable(version, byref(_Driver)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Driver = _Driver

        # attach function interface to function address
        self.zesDriverEventListen = _zesDriverEventListen_t(self.__dditable.Driver.pfnEventListen)
        self.zesDriverEventListenEx = _zesDriverEventListenEx_t(self.__dditable.Driver.pfnEventListenEx)

        # call driver to get function pointers
        _Device = _zes_device_dditable_t()
        r = ze_result_v(self.__dll.zesGetDeviceProcAddrTable(version, byref(_Device)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Device = _Device

        # attach function interface to function address
        self.zesDeviceGetProperties = _zesDeviceGetProperties_t(self.__dditable.Device.pfnGetProperties)
        self.zesDeviceGetState = _zesDeviceGetState_t(self.__dditable.Device.pfnGetState)
        self.zesDeviceReset = _zesDeviceReset_t(self.__dditable.Device.pfnReset)
        self.zesDeviceProcessesGetState = _zesDeviceProcessesGetState_t(self.__dditable.Device.pfnProcessesGetState)
        self.zesDevicePciGetProperties = _zesDevicePciGetProperties_t(self.__dditable.Device.pfnPciGetProperties)
        self.zesDevicePciGetState = _zesDevicePciGetState_t(self.__dditable.Device.pfnPciGetState)
        self.zesDevicePciGetBars = _zesDevicePciGetBars_t(self.__dditable.Device.pfnPciGetBars)
        self.zesDevicePciGetStats = _zesDevicePciGetStats_t(self.__dditable.Device.pfnPciGetStats)
        self.zesDeviceEnumDiagnosticTestSuites = _zesDeviceEnumDiagnosticTestSuites_t(self.__dditable.Device.pfnEnumDiagnosticTestSuites)
        self.zesDeviceEnumEngineGroups = _zesDeviceEnumEngineGroups_t(self.__dditable.Device.pfnEnumEngineGroups)
        self.zesDeviceEventRegister = _zesDeviceEventRegister_t(self.__dditable.Device.pfnEventRegister)
        self.zesDeviceEnumFabricPorts = _zesDeviceEnumFabricPorts_t(self.__dditable.Device.pfnEnumFabricPorts)
        self.zesDeviceEnumFans = _zesDeviceEnumFans_t(self.__dditable.Device.pfnEnumFans)
        self.zesDeviceEnumFirmwares = _zesDeviceEnumFirmwares_t(self.__dditable.Device.pfnEnumFirmwares)
        self.zesDeviceEnumFrequencyDomains = _zesDeviceEnumFrequencyDomains_t(self.__dditable.Device.pfnEnumFrequencyDomains)
        self.zesDeviceEnumLeds = _zesDeviceEnumLeds_t(self.__dditable.Device.pfnEnumLeds)
        self.zesDeviceEnumMemoryModules = _zesDeviceEnumMemoryModules_t(self.__dditable.Device.pfnEnumMemoryModules)
        self.zesDeviceEnumPerformanceFactorDomains = _zesDeviceEnumPerformanceFactorDomains_t(self.__dditable.Device.pfnEnumPerformanceFactorDomains)
        self.zesDeviceEnumPowerDomains = _zesDeviceEnumPowerDomains_t(self.__dditable.Device.pfnEnumPowerDomains)
        self.zesDeviceGetCardPowerDomain = _zesDeviceGetCardPowerDomain_t(self.__dditable.Device.pfnGetCardPowerDomain)
        self.zesDeviceEnumPsus = _zesDeviceEnumPsus_t(self.__dditable.Device.pfnEnumPsus)
        self.zesDeviceEnumRasErrorSets = _zesDeviceEnumRasErrorSets_t(self.__dditable.Device.pfnEnumRasErrorSets)
        self.zesDeviceEnumSchedulers = _zesDeviceEnumSchedulers_t(self.__dditable.Device.pfnEnumSchedulers)
        self.zesDeviceEnumStandbyDomains = _zesDeviceEnumStandbyDomains_t(self.__dditable.Device.pfnEnumStandbyDomains)
        self.zesDeviceEnumTemperatureSensors = _zesDeviceEnumTemperatureSensors_t(self.__dditable.Device.pfnEnumTemperatureSensors)

        # call driver to get function pointers
        _Scheduler = _zes_scheduler_dditable_t()
        r = ze_result_v(self.__dll.zesGetSchedulerProcAddrTable(version, byref(_Scheduler)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Scheduler = _Scheduler

        # attach function interface to function address
        self.zesSchedulerGetProperties = _zesSchedulerGetProperties_t(self.__dditable.Scheduler.pfnGetProperties)
        self.zesSchedulerGetCurrentMode = _zesSchedulerGetCurrentMode_t(self.__dditable.Scheduler.pfnGetCurrentMode)
        self.zesSchedulerGetTimeoutModeProperties = _zesSchedulerGetTimeoutModeProperties_t(self.__dditable.Scheduler.pfnGetTimeoutModeProperties)
        self.zesSchedulerGetTimesliceModeProperties = _zesSchedulerGetTimesliceModeProperties_t(self.__dditable.Scheduler.pfnGetTimesliceModeProperties)
        self.zesSchedulerSetTimeoutMode = _zesSchedulerSetTimeoutMode_t(self.__dditable.Scheduler.pfnSetTimeoutMode)
        self.zesSchedulerSetTimesliceMode = _zesSchedulerSetTimesliceMode_t(self.__dditable.Scheduler.pfnSetTimesliceMode)
        self.zesSchedulerSetExclusiveMode = _zesSchedulerSetExclusiveMode_t(self.__dditable.Scheduler.pfnSetExclusiveMode)
        self.zesSchedulerSetComputeUnitDebugMode = _zesSchedulerSetComputeUnitDebugMode_t(self.__dditable.Scheduler.pfnSetComputeUnitDebugMode)

        # call driver to get function pointers
        _PerformanceFactor = _zes_performance_factor_dditable_t()
        r = ze_result_v(self.__dll.zesGetPerformanceFactorProcAddrTable(version, byref(_PerformanceFactor)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.PerformanceFactor = _PerformanceFactor

        # attach function interface to function address
        self.zesPerformanceFactorGetProperties = _zesPerformanceFactorGetProperties_t(self.__dditable.PerformanceFactor.pfnGetProperties)
        self.zesPerformanceFactorGetConfig = _zesPerformanceFactorGetConfig_t(self.__dditable.PerformanceFactor.pfnGetConfig)
        self.zesPerformanceFactorSetConfig = _zesPerformanceFactorSetConfig_t(self.__dditable.PerformanceFactor.pfnSetConfig)

        # call driver to get function pointers
        _Power = _zes_power_dditable_t()
        r = ze_result_v(self.__dll.zesGetPowerProcAddrTable(version, byref(_Power)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Power = _Power

        # attach function interface to function address
        self.zesPowerGetProperties = _zesPowerGetProperties_t(self.__dditable.Power.pfnGetProperties)
        self.zesPowerGetEnergyCounter = _zesPowerGetEnergyCounter_t(self.__dditable.Power.pfnGetEnergyCounter)
        self.zesPowerGetLimits = _zesPowerGetLimits_t(self.__dditable.Power.pfnGetLimits)
        self.zesPowerSetLimits = _zesPowerSetLimits_t(self.__dditable.Power.pfnSetLimits)
        self.zesPowerGetEnergyThreshold = _zesPowerGetEnergyThreshold_t(self.__dditable.Power.pfnGetEnergyThreshold)
        self.zesPowerSetEnergyThreshold = _zesPowerSetEnergyThreshold_t(self.__dditable.Power.pfnSetEnergyThreshold)

        # call driver to get function pointers
        _Frequency = _zes_frequency_dditable_t()
        r = ze_result_v(self.__dll.zesGetFrequencyProcAddrTable(version, byref(_Frequency)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Frequency = _Frequency

        # attach function interface to function address
        self.zesFrequencyGetProperties = _zesFrequencyGetProperties_t(self.__dditable.Frequency.pfnGetProperties)
        self.zesFrequencyGetAvailableClocks = _zesFrequencyGetAvailableClocks_t(self.__dditable.Frequency.pfnGetAvailableClocks)
        self.zesFrequencyGetRange = _zesFrequencyGetRange_t(self.__dditable.Frequency.pfnGetRange)
        self.zesFrequencySetRange = _zesFrequencySetRange_t(self.__dditable.Frequency.pfnSetRange)
        self.zesFrequencyGetState = _zesFrequencyGetState_t(self.__dditable.Frequency.pfnGetState)
        self.zesFrequencyGetThrottleTime = _zesFrequencyGetThrottleTime_t(self.__dditable.Frequency.pfnGetThrottleTime)
        self.zesFrequencyOcGetCapabilities = _zesFrequencyOcGetCapabilities_t(self.__dditable.Frequency.pfnOcGetCapabilities)
        self.zesFrequencyOcGetFrequencyTarget = _zesFrequencyOcGetFrequencyTarget_t(self.__dditable.Frequency.pfnOcGetFrequencyTarget)
        self.zesFrequencyOcSetFrequencyTarget = _zesFrequencyOcSetFrequencyTarget_t(self.__dditable.Frequency.pfnOcSetFrequencyTarget)
        self.zesFrequencyOcGetVoltageTarget = _zesFrequencyOcGetVoltageTarget_t(self.__dditable.Frequency.pfnOcGetVoltageTarget)
        self.zesFrequencyOcSetVoltageTarget = _zesFrequencyOcSetVoltageTarget_t(self.__dditable.Frequency.pfnOcSetVoltageTarget)
        self.zesFrequencyOcSetMode = _zesFrequencyOcSetMode_t(self.__dditable.Frequency.pfnOcSetMode)
        self.zesFrequencyOcGetMode = _zesFrequencyOcGetMode_t(self.__dditable.Frequency.pfnOcGetMode)
        self.zesFrequencyOcGetIccMax = _zesFrequencyOcGetIccMax_t(self.__dditable.Frequency.pfnOcGetIccMax)
        self.zesFrequencyOcSetIccMax = _zesFrequencyOcSetIccMax_t(self.__dditable.Frequency.pfnOcSetIccMax)
        self.zesFrequencyOcGetTjMax = _zesFrequencyOcGetTjMax_t(self.__dditable.Frequency.pfnOcGetTjMax)
        self.zesFrequencyOcSetTjMax = _zesFrequencyOcSetTjMax_t(self.__dditable.Frequency.pfnOcSetTjMax)

        # call driver to get function pointers
        _Engine = _zes_engine_dditable_t()
        r = ze_result_v(self.__dll.zesGetEngineProcAddrTable(version, byref(_Engine)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Engine = _Engine

        # attach function interface to function address
        self.zesEngineGetProperties = _zesEngineGetProperties_t(self.__dditable.Engine.pfnGetProperties)
        self.zesEngineGetActivity = _zesEngineGetActivity_t(self.__dditable.Engine.pfnGetActivity)

        # call driver to get function pointers
        _Standby = _zes_standby_dditable_t()
        r = ze_result_v(self.__dll.zesGetStandbyProcAddrTable(version, byref(_Standby)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Standby = _Standby

        # attach function interface to function address
        self.zesStandbyGetProperties = _zesStandbyGetProperties_t(self.__dditable.Standby.pfnGetProperties)
        self.zesStandbyGetMode = _zesStandbyGetMode_t(self.__dditable.Standby.pfnGetMode)
        self.zesStandbySetMode = _zesStandbySetMode_t(self.__dditable.Standby.pfnSetMode)

        # call driver to get function pointers
        _Firmware = _zes_firmware_dditable_t()
        r = ze_result_v(self.__dll.zesGetFirmwareProcAddrTable(version, byref(_Firmware)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Firmware = _Firmware

        # attach function interface to function address
        self.zesFirmwareGetProperties = _zesFirmwareGetProperties_t(self.__dditable.Firmware.pfnGetProperties)
        self.zesFirmwareFlash = _zesFirmwareFlash_t(self.__dditable.Firmware.pfnFlash)

        # call driver to get function pointers
        _Memory = _zes_memory_dditable_t()
        r = ze_result_v(self.__dll.zesGetMemoryProcAddrTable(version, byref(_Memory)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Memory = _Memory

        # attach function interface to function address
        self.zesMemoryGetProperties = _zesMemoryGetProperties_t(self.__dditable.Memory.pfnGetProperties)
        self.zesMemoryGetState = _zesMemoryGetState_t(self.__dditable.Memory.pfnGetState)
        self.zesMemoryGetBandwidth = _zesMemoryGetBandwidth_t(self.__dditable.Memory.pfnGetBandwidth)

        # call driver to get function pointers
        _FabricPort = _zes_fabric_port_dditable_t()
        r = ze_result_v(self.__dll.zesGetFabricPortProcAddrTable(version, byref(_FabricPort)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.FabricPort = _FabricPort

        # attach function interface to function address
        self.zesFabricPortGetProperties = _zesFabricPortGetProperties_t(self.__dditable.FabricPort.pfnGetProperties)
        self.zesFabricPortGetLinkType = _zesFabricPortGetLinkType_t(self.__dditable.FabricPort.pfnGetLinkType)
        self.zesFabricPortGetConfig = _zesFabricPortGetConfig_t(self.__dditable.FabricPort.pfnGetConfig)
        self.zesFabricPortSetConfig = _zesFabricPortSetConfig_t(self.__dditable.FabricPort.pfnSetConfig)
        self.zesFabricPortGetState = _zesFabricPortGetState_t(self.__dditable.FabricPort.pfnGetState)
        self.zesFabricPortGetThroughput = _zesFabricPortGetThroughput_t(self.__dditable.FabricPort.pfnGetThroughput)

        # call driver to get function pointers
        _Temperature = _zes_temperature_dditable_t()
        r = ze_result_v(self.__dll.zesGetTemperatureProcAddrTable(version, byref(_Temperature)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Temperature = _Temperature

        # attach function interface to function address
        self.zesTemperatureGetProperties = _zesTemperatureGetProperties_t(self.__dditable.Temperature.pfnGetProperties)
        self.zesTemperatureGetConfig = _zesTemperatureGetConfig_t(self.__dditable.Temperature.pfnGetConfig)
        self.zesTemperatureSetConfig = _zesTemperatureSetConfig_t(self.__dditable.Temperature.pfnSetConfig)
        self.zesTemperatureGetState = _zesTemperatureGetState_t(self.__dditable.Temperature.pfnGetState)

        # call driver to get function pointers
        _Psu = _zes_psu_dditable_t()
        r = ze_result_v(self.__dll.zesGetPsuProcAddrTable(version, byref(_Psu)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Psu = _Psu

        # attach function interface to function address
        self.zesPsuGetProperties = _zesPsuGetProperties_t(self.__dditable.Psu.pfnGetProperties)
        self.zesPsuGetState = _zesPsuGetState_t(self.__dditable.Psu.pfnGetState)

        # call driver to get function pointers
        _Fan = _zes_fan_dditable_t()
        r = ze_result_v(self.__dll.zesGetFanProcAddrTable(version, byref(_Fan)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Fan = _Fan

        # attach function interface to function address
        self.zesFanGetProperties = _zesFanGetProperties_t(self.__dditable.Fan.pfnGetProperties)
        self.zesFanGetConfig = _zesFanGetConfig_t(self.__dditable.Fan.pfnGetConfig)
        self.zesFanSetDefaultMode = _zesFanSetDefaultMode_t(self.__dditable.Fan.pfnSetDefaultMode)
        self.zesFanSetFixedSpeedMode = _zesFanSetFixedSpeedMode_t(self.__dditable.Fan.pfnSetFixedSpeedMode)
        self.zesFanSetSpeedTableMode = _zesFanSetSpeedTableMode_t(self.__dditable.Fan.pfnSetSpeedTableMode)
        self.zesFanGetState = _zesFanGetState_t(self.__dditable.Fan.pfnGetState)

        # call driver to get function pointers
        _Led = _zes_led_dditable_t()
        r = ze_result_v(self.__dll.zesGetLedProcAddrTable(version, byref(_Led)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Led = _Led

        # attach function interface to function address
        self.zesLedGetProperties = _zesLedGetProperties_t(self.__dditable.Led.pfnGetProperties)
        self.zesLedGetState = _zesLedGetState_t(self.__dditable.Led.pfnGetState)
        self.zesLedSetState = _zesLedSetState_t(self.__dditable.Led.pfnSetState)
        self.zesLedSetColor = _zesLedSetColor_t(self.__dditable.Led.pfnSetColor)

        # call driver to get function pointers
        _Ras = _zes_ras_dditable_t()
        r = ze_result_v(self.__dll.zesGetRasProcAddrTable(version, byref(_Ras)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Ras = _Ras

        # attach function interface to function address
        self.zesRasGetProperties = _zesRasGetProperties_t(self.__dditable.Ras.pfnGetProperties)
        self.zesRasGetConfig = _zesRasGetConfig_t(self.__dditable.Ras.pfnGetConfig)
        self.zesRasSetConfig = _zesRasSetConfig_t(self.__dditable.Ras.pfnSetConfig)
        self.zesRasGetState = _zesRasGetState_t(self.__dditable.Ras.pfnGetState)

        # call driver to get function pointers
        _Diagnostics = _zes_diagnostics_dditable_t()
        r = ze_result_v(self.__dll.zesGetDiagnosticsProcAddrTable(version, byref(_Diagnostics)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Diagnostics = _Diagnostics

        # attach function interface to function address
        self.zesDiagnosticsGetProperties = _zesDiagnosticsGetProperties_t(self.__dditable.Diagnostics.pfnGetProperties)
        self.zesDiagnosticsGetTests = _zesDiagnosticsGetTests_t(self.__dditable.Diagnostics.pfnGetTests)
        self.zesDiagnosticsRunTests = _zesDiagnosticsRunTests_t(self.__dditable.Diagnostics.pfnRunTests)

        # success!
