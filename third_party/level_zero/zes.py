"""
 Copyright (C) 2019-2021 Intel Corporation

 SPDX-License-Identifier: MIT

 @file zes.py
 @version v1.11-r1.11.8

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
## @brief Handle for a Sysman device overclock domain
class zes_overclock_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle for a Sysman virtual function management domain
class zes_vf_handle_t(c_void_p):
    pass

###############################################################################
## @brief Defines structure types
class zes_structure_type_v(IntEnum):
    DEVICE_PROPERTIES = 0x1                                                 ## ::zes_device_properties_t
    PCI_PROPERTIES = 0x2                                                    ## ::zes_pci_properties_t
    PCI_BAR_PROPERTIES = 0x3                                                ## ::zes_pci_bar_properties_t
    DIAG_PROPERTIES = 0x4                                                   ## ::zes_diag_properties_t
    ENGINE_PROPERTIES = 0x5                                                 ## ::zes_engine_properties_t
    FABRIC_PORT_PROPERTIES = 0x6                                            ## ::zes_fabric_port_properties_t
    FAN_PROPERTIES = 0x7                                                    ## ::zes_fan_properties_t
    FIRMWARE_PROPERTIES = 0x8                                               ## ::zes_firmware_properties_t
    FREQ_PROPERTIES = 0x9                                                   ## ::zes_freq_properties_t
    LED_PROPERTIES = 0xa                                                    ## ::zes_led_properties_t
    MEM_PROPERTIES = 0xb                                                    ## ::zes_mem_properties_t
    PERF_PROPERTIES = 0xc                                                   ## ::zes_perf_properties_t
    POWER_PROPERTIES = 0xd                                                  ## ::zes_power_properties_t
    PSU_PROPERTIES = 0xe                                                    ## ::zes_psu_properties_t
    RAS_PROPERTIES = 0xf                                                    ## ::zes_ras_properties_t
    SCHED_PROPERTIES = 0x10                                                 ## ::zes_sched_properties_t
    SCHED_TIMEOUT_PROPERTIES = 0x11                                         ## ::zes_sched_timeout_properties_t
    SCHED_TIMESLICE_PROPERTIES = 0x12                                       ## ::zes_sched_timeslice_properties_t
    STANDBY_PROPERTIES = 0x13                                               ## ::zes_standby_properties_t
    TEMP_PROPERTIES = 0x14                                                  ## ::zes_temp_properties_t
    DEVICE_STATE = 0x15                                                     ## ::zes_device_state_t
    PROCESS_STATE = 0x16                                                    ## ::zes_process_state_t
    PCI_STATE = 0x17                                                        ## ::zes_pci_state_t
    FABRIC_PORT_CONFIG = 0x18                                               ## ::zes_fabric_port_config_t
    FABRIC_PORT_STATE = 0x19                                                ## ::zes_fabric_port_state_t
    FAN_CONFIG = 0x1a                                                       ## ::zes_fan_config_t
    FREQ_STATE = 0x1b                                                       ## ::zes_freq_state_t
    OC_CAPABILITIES = 0x1c                                                  ## ::zes_oc_capabilities_t
    LED_STATE = 0x1d                                                        ## ::zes_led_state_t
    MEM_STATE = 0x1e                                                        ## ::zes_mem_state_t
    PSU_STATE = 0x1f                                                        ## ::zes_psu_state_t
    BASE_STATE = 0x20                                                       ## ::zes_base_state_t
    RAS_CONFIG = 0x21                                                       ## ::zes_ras_config_t
    RAS_STATE = 0x22                                                        ## ::zes_ras_state_t
    TEMP_CONFIG = 0x23                                                      ## ::zes_temp_config_t
    PCI_BAR_PROPERTIES_1_2 = 0x24                                           ## ::zes_pci_bar_properties_1_2_t
    DEVICE_ECC_DESC = 0x25                                                  ## ::zes_device_ecc_desc_t
    DEVICE_ECC_PROPERTIES = 0x26                                            ## ::zes_device_ecc_properties_t
    POWER_LIMIT_EXT_DESC = 0x27                                             ## ::zes_power_limit_ext_desc_t
    POWER_EXT_PROPERTIES = 0x28                                             ## ::zes_power_ext_properties_t
    OVERCLOCK_PROPERTIES = 0x29                                             ## ::zes_overclock_properties_t
    FABRIC_PORT_ERROR_COUNTERS = 0x2a                                       ## ::zes_fabric_port_error_counters_t
    ENGINE_EXT_PROPERTIES = 0x2b                                            ## ::zes_engine_ext_properties_t
    RESET_PROPERTIES = 0x2c                                                 ## ::zes_reset_properties_t
    DEVICE_EXT_PROPERTIES = 0x2d                                            ## ::zes_device_ext_properties_t
    DEVICE_UUID = 0x2e                                                      ## ::zes_uuid_t
    POWER_DOMAIN_EXP_PROPERTIES = 0x00020001                                ## ::zes_power_domain_exp_properties_t
    MEM_BANDWIDTH_COUNTER_BITS_EXP_PROPERTIES = 0x00020002                  ## ::zes_mem_bandwidth_counter_bits_exp_properties_t
    MEMORY_PAGE_OFFLINE_STATE_EXP = 0x00020003                              ## ::zes_mem_page_offline_state_exp_t
    SUBDEVICE_EXP_PROPERTIES = 0x00020004                                   ## ::zes_subdevice_exp_properties_t
    VF_EXP_PROPERTIES = 0x00020005                                          ## ::zes_vf_exp_properties_t
    VF_UTIL_MEM_EXP = 0x00020006                                            ## ::zes_vf_util_mem_exp_t
    VF_UTIL_ENGINE_EXP = 0x00020007                                         ## ::zes_vf_util_engine_exp_t
    VF_EXP_CAPABILITIES = 0x00020008                                        ## ::zes_vf_exp_capabilities_t
    VF_UTIL_MEM_EXP2 = 0x00020009                                           ## ::zes_vf_util_mem_exp2_t
    VF_UTIL_ENGINE_EXP2 = 0x00020010                                        ## ::zes_vf_util_engine_exp2_t

class zes_structure_type_t(c_int):
    def __str__(self):
        return str(zes_structure_type_v(self.value))


###############################################################################
## @brief Base for all properties types
class zes_base_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p)                                             ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
    ]

###############################################################################
## @brief Base for all descriptor types
class zes_base_desc_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p)                                             ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
    ]

###############################################################################
## @brief Base for all state types
class zes_base_state_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p)                                             ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
    ]

###############################################################################
## @brief Base for all config types
class zes_base_config_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p)                                             ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
    ]

###############################################################################
## @brief Base for all capability types
class zes_base_capability_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p)                                             ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
    ]

###############################################################################
## @brief Supported sysman initialization flags
class zes_init_flags_v(IntEnum):
    PLACEHOLDER = ZE_BIT(0)                                                 ## placeholder for future use

class zes_init_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Maximum extension name string size
ZES_MAX_EXTENSION_NAME = 256

###############################################################################
## @brief Extension properties queried using ::zesDriverGetExtensionProperties
class zes_driver_extension_properties_t(Structure):
    _fields_ = [
        ("name", c_char * ZES_MAX_EXTENSION_NAME),                      ## [out] extension name
        ("version", c_ulong)                                            ## [out] extension version using ::ZE_MAKE_VERSION
    ]

###############################################################################
## @brief Maximum number of characters in string properties.
ZES_STRING_PROPERTY_SIZE = 64

###############################################################################
## @brief Maximum device universal unique id (UUID) size in bytes.
ZES_MAX_UUID_SIZE = 16

###############################################################################
## @brief Types of accelerator engines
class zes_engine_type_flags_v(IntEnum):
    OTHER = ZE_BIT(0)                                                       ## Undefined types of accelerators.
    COMPUTE = ZE_BIT(1)                                                     ## Engines that process compute kernels only (no 3D content).
    _3D = ZE_BIT(2)                                                         ## Engines that process 3D content only (no compute kernels).
    MEDIA = ZE_BIT(3)                                                       ## Engines that process media workloads.
    DMA = ZE_BIT(4)                                                         ## Engines that copy blocks of data.
    RENDER = ZE_BIT(5)                                                      ## Engines that can process both 3D content and compute kernels.

class zes_engine_type_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Device repair status
class zes_repair_status_v(IntEnum):
    UNSUPPORTED = 0                                                         ## The device does not support in-field repairs.
    NOT_PERFORMED = 1                                                       ## The device has never been repaired.
    PERFORMED = 2                                                           ## The device has been repaired.

class zes_repair_status_t(c_int):
    def __str__(self):
        return str(zes_repair_status_v(self.value))


###############################################################################
## @brief Device reset reasons
class zes_reset_reason_flags_v(IntEnum):
    WEDGED = ZE_BIT(0)                                                      ## The device needs to be reset because one or more parts of the hardware
                                                                            ## is wedged
    REPAIR = ZE_BIT(1)                                                      ## The device needs to be reset in order to complete in-field repairs

class zes_reset_reason_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Device reset type
class zes_reset_type_v(IntEnum):
    WARM = 0                                                                ## Apply warm reset
    COLD = 1                                                                ## Apply cold reset
    FLR = 2                                                                 ## Apply FLR reset

class zes_reset_type_t(c_int):
    def __str__(self):
        return str(zes_reset_type_v(self.value))


###############################################################################
## @brief Device state
class zes_device_state_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("reset", zes_reset_reason_flags_t),                            ## [out] Indicates if the device needs to be reset and for what reasons.
                                                                        ## returns 0 (none) or combination of ::zes_reset_reason_flag_t
        ("repaired", zes_repair_status_t)                               ## [out] Indicates if the device has been repaired
    ]

###############################################################################
## @brief Device reset properties
class zes_reset_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("force", ze_bool_t),                                           ## [in] If set to true, all applications that are currently using the
                                                                        ## device will be forcibly killed.
        ("resetType", zes_reset_type_t)                                 ## [in] Type of reset needs to be performed
    ]

###############################################################################
## @brief Device universal unique id (UUID)
class zes_uuid_t(Structure):
    _fields_ = [
        ("id", c_ubyte * ZES_MAX_UUID_SIZE)                             ## [out] opaque data representing a device UUID
    ]

###############################################################################
## @brief Supported device types
class zes_device_type_v(IntEnum):
    GPU = 1                                                                 ## Graphics Processing Unit
    CPU = 2                                                                 ## Central Processing Unit
    FPGA = 3                                                                ## Field Programmable Gate Array
    MCA = 4                                                                 ## Memory Copy Accelerator
    VPU = 5                                                                 ## Vision Processing Unit

class zes_device_type_t(c_int):
    def __str__(self):
        return str(zes_device_type_v(self.value))


###############################################################################
## @brief Supported device property flags
class zes_device_property_flags_v(IntEnum):
    INTEGRATED = ZE_BIT(0)                                                  ## Device is integrated with the Host.
    SUBDEVICE = ZE_BIT(1)                                                   ## Device handle used for query represents a sub-device.
    ECC = ZE_BIT(2)                                                         ## Device supports error correction memory access.
    ONDEMANDPAGING = ZE_BIT(3)                                              ## Device supports on-demand page-faulting.

class zes_device_property_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Device properties
class zes_device_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("core", ze_device_properties_t),                               ## [out] (Deprecated, use ::zes_uuid_t in the extended structure) Core
                                                                        ## device properties
        ("numSubdevices", c_ulong),                                     ## [out] Number of sub-devices. A value of 0 indicates that this device
                                                                        ## doesn't have sub-devices.
        ("serialNumber", c_char * ZES_STRING_PROPERTY_SIZE),            ## [out] Manufacturing serial number (NULL terminated string value). This
                                                                        ## value is intended to reflect the Part ID/SoC ID assigned by
                                                                        ## manufacturer that is unique for a SoC. Will be set to the string
                                                                        ## "unknown" if this cannot be determined for the device.
        ("boardNumber", c_char * ZES_STRING_PROPERTY_SIZE),             ## [out] Manufacturing board number (NULL terminated string value).
                                                                        ## Alternatively "boardSerialNumber", this value is intended to reflect
                                                                        ## the string printed on board label by manufacturer. Will be set to the
                                                                        ## string "unknown" if this cannot be determined for the device.
        ("brandName", c_char * ZES_STRING_PROPERTY_SIZE),               ## [out] Brand name of the device (NULL terminated string value). Will be
                                                                        ## set to the string "unknown" if this cannot be determined for the
                                                                        ## device.
        ("modelName", c_char * ZES_STRING_PROPERTY_SIZE),               ## [out] Model name of the device (NULL terminated string value). Will be
                                                                        ## set to the string "unknown" if this cannot be determined for the
                                                                        ## device.
        ("vendorName", c_char * ZES_STRING_PROPERTY_SIZE),              ## [out] Vendor name of the device (NULL terminated string value). Will
                                                                        ## be set to the string "unknown" if this cannot be determined for the
                                                                        ## device.
        ("driverVersion", c_char * ZES_STRING_PROPERTY_SIZE)            ## [out] Installed driver version (NULL terminated string value). Will be
                                                                        ## set to the string "unknown" if this cannot be determined for the
                                                                        ## device.
    ]

###############################################################################
## @brief Device properties
class zes_device_ext_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("uuid", zes_uuid_t),                                           ## [out] universal unique identifier. Note: uuid obtained from Sysman API
                                                                        ## is the same as from core API. Subdevices will have their own uuid.
        ("type", zes_device_type_t),                                    ## [out] generic device type
        ("flags", zes_device_property_flags_t)                          ## [out] 0 (none) or a valid combination of ::zes_device_property_flag_t
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
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("address", zes_pci_address_t),                                 ## [out] The BDF address
        ("maxSpeed", zes_pci_speed_t),                                  ## [out] Fastest port configuration supported by the device (sum of all
                                                                        ## lanes)
        ("haveBandwidthCounters", ze_bool_t),                           ## [out] Indicates whether the `rxCounter` and `txCounter` members of
                                                                        ## ::zes_pci_stats_t will have valid values
        ("havePacketCounters", ze_bool_t),                              ## [out] Indicates whether the `packetCounter` member of
                                                                        ## ::zes_pci_stats_t will have a valid value
        ("haveReplayCounters", ze_bool_t)                               ## [out] Indicates whether the `replayCounter` member of
                                                                        ## ::zes_pci_stats_t will have a valid value
    ]

###############################################################################
## @brief PCI link status
class zes_pci_link_status_v(IntEnum):
    UNKNOWN = 0                                                             ## The link status could not be determined
    GOOD = 1                                                                ## The link is up and operating as expected
    QUALITY_ISSUES = 2                                                      ## The link is up but has quality and/or bandwidth degradation
    STABILITY_ISSUES = 3                                                    ## The link has stability issues and preventing workloads making forward
                                                                            ## progress

class zes_pci_link_status_t(c_int):
    def __str__(self):
        return str(zes_pci_link_status_v(self.value))


###############################################################################
## @brief PCI link quality degradation reasons
class zes_pci_link_qual_issue_flags_v(IntEnum):
    REPLAYS = ZE_BIT(0)                                                     ## A significant number of replays are occurring
    SPEED = ZE_BIT(1)                                                       ## There is a degradation in the maximum bandwidth of the link

class zes_pci_link_qual_issue_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief PCI link stability issues
class zes_pci_link_stab_issue_flags_v(IntEnum):
    RETRAINING = ZE_BIT(0)                                                  ## Link retraining has occurred to deal with quality issues

class zes_pci_link_stab_issue_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Dynamic PCI state
class zes_pci_state_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
    MMIO = 0                                                                ## MMIO registers
    ROM = 1                                                                 ## ROM aperture
    MEM = 2                                                                 ## Device memory

class zes_pci_bar_type_t(c_int):
    def __str__(self):
        return str(zes_pci_bar_type_v(self.value))


###############################################################################
## @brief Properties of a pci bar
class zes_pci_bar_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
                                                                        ## lanes). Will always be 0 when the `haveReplayCounters` member of
                                                                        ## ::zes_pci_properties_t is FALSE.
        ("packetCounter", c_ulonglong),                                 ## [out] Monotonic counter for the number of packets (sum of all lanes).
                                                                        ## Will always be 0 when the `havePacketCounters` member of
                                                                        ## ::zes_pci_properties_t is FALSE.
        ("rxCounter", c_ulonglong),                                     ## [out] Monotonic counter for the number of bytes received (sum of all
                                                                        ## lanes). Will always be 0 when the `haveBandwidthCounters` member of
                                                                        ## ::zes_pci_properties_t is FALSE.
        ("txCounter", c_ulonglong),                                     ## [out] Monotonic counter for the number of bytes transmitted (including
                                                                        ## replays) (sum of all lanes). Will always be 0 when the
                                                                        ## `haveBandwidthCounters` member of ::zes_pci_properties_t is FALSE.
        ("speed", zes_pci_speed_t)                                      ## [out] The current speed of the link (sum of all lanes)
    ]

###############################################################################
## @brief Overclock domains.
class zes_overclock_domain_v(IntEnum):
    CARD = 1                                                                ## Overclocking card level properties such as temperature limits.
    PACKAGE = 2                                                             ## Overclocking package level properties such as power limits.
    GPU_ALL = 4                                                             ## Overclocking a GPU that has all accelerator assets on the same PLL/VR.
    GPU_RENDER_COMPUTE = 8                                                  ## Overclocking a GPU with render and compute assets on the same PLL/VR.
    GPU_RENDER = 16                                                         ## Overclocking a GPU with render assets on its own PLL/VR.
    GPU_COMPUTE = 32                                                        ## Overclocking a GPU with compute assets on its own PLL/VR.
    GPU_MEDIA = 64                                                          ## Overclocking a GPU with media assets on its own PLL/VR.
    VRAM = 128                                                              ## Overclocking device local memory.
    ADM = 256                                                               ## Overclocking LLC/L4 cache.

class zes_overclock_domain_t(c_int):
    def __str__(self):
        return str(zes_overclock_domain_v(self.value))


###############################################################################
## @brief Overclock controls.
class zes_overclock_control_v(IntEnum):
    VF = 1                                                                  ## This control permits setting a custom V-F curve.
    FREQ_OFFSET = 2                                                         ## The V-F curve of the overclock domain can be shifted up or down using
                                                                            ## this control.
    VMAX_OFFSET = 4                                                         ## This control is used to increase the permitted voltage above the
                                                                            ## shipped voltage maximum.
    FREQ = 8                                                                ## This control permits direct changes to the operating frequency.
    VOLT_LIMIT = 16                                                         ## This control prevents frequencies that would push the voltage above
                                                                            ## this value, typically used by V-F scanners.
    POWER_SUSTAINED_LIMIT = 32                                              ## This control changes the sustained power limit (PL1).
    POWER_BURST_LIMIT = 64                                                  ## This control changes the burst power limit (PL2).
    POWER_PEAK_LIMIT = 128                                                  ## his control changes the peak power limit (PL4).
    ICCMAX_LIMIT = 256                                                      ## This control changes the value of IccMax..
    TEMP_LIMIT = 512                                                        ## This control changes the value of TjMax.
    ITD_DISABLE = 1024                                                      ## This control permits disabling the adaptive voltage feature ITD
    ACM_DISABLE = 2048                                                      ## This control permits disabling the adaptive voltage feature ACM.

class zes_overclock_control_t(c_int):
    def __str__(self):
        return str(zes_overclock_control_v(self.value))


###############################################################################
## @brief Overclock modes.
class zes_overclock_mode_v(IntEnum):
    MODE_OFF = 0                                                            ## Overclock mode is off
    MODE_STOCK = 2                                                          ## Stock (manufacturing settings) are being used.
    MODE_ON = 3                                                             ## Overclock mode is on.
    MODE_UNAVAILABLE = 4                                                    ## Overclocking is unavailable at this time since the system is running
                                                                            ## on battery.
    MODE_DISABLED = 5                                                       ## Overclock mode is disabled.

class zes_overclock_mode_t(c_int):
    def __str__(self):
        return str(zes_overclock_mode_v(self.value))


###############################################################################
## @brief Overclock control states.
class zes_control_state_v(IntEnum):
    STATE_UNSET = 0                                                         ## No overclock control has not been changed by the driver since the last
                                                                            ## boot/reset.
    STATE_ACTIVE = 2                                                        ## The overclock control has been set and it is active.
    STATE_DISABLED = 3                                                      ## The overclock control value has been disabled due to the current power
                                                                            ## configuration (typically when running on DC).

class zes_control_state_t(c_int):
    def __str__(self):
        return str(zes_control_state_v(self.value))


###############################################################################
## @brief Overclock pending actions.
class zes_pending_action_v(IntEnum):
    PENDING_NONE = 0                                                        ## There no pending actions. .
    PENDING_IMMINENT = 1                                                    ## The requested change is in progress and should complete soon.
    PENDING_COLD_RESET = 2                                                  ## The requested change requires a device cold reset (hotplug, system
                                                                            ## boot).
    PENDING_WARM_RESET = 3                                                  ## The requested change requires a device warm reset (PCIe FLR).

class zes_pending_action_t(c_int):
    def __str__(self):
        return str(zes_pending_action_v(self.value))


###############################################################################
## @brief Overclock V-F curve programing.
class zes_vf_program_type_v(IntEnum):
    VF_ARBITRARY = 0                                                        ## Can program an arbitrary number of V-F points up to the maximum number
                                                                            ## and each point can have arbitrary voltage and frequency values within
                                                                            ## the min/max/step limits
    VF_FREQ_FIXED = 1                                                       ## Can only program the voltage for the V-F points that it reads back -
                                                                            ## the frequency of those points cannot be changed
    VF_VOLT_FIXED = 2                                                       ## Can only program the frequency for the V-F points that is reads back -
                                                                            ## the voltage of each point cannot be changed.

class zes_vf_program_type_t(c_int):
    def __str__(self):
        return str(zes_vf_program_type_v(self.value))


###############################################################################
## @brief VF type
class zes_vf_type_v(IntEnum):
    VOLT = 0                                                                ## VF Voltage point
    FREQ = 1                                                                ## VF Frequency point

class zes_vf_type_t(c_int):
    def __str__(self):
        return str(zes_vf_type_v(self.value))


###############################################################################
## @brief VF type
class zes_vf_array_type_v(IntEnum):
    USER_VF_ARRAY = 0                                                       ## User V-F array
    DEFAULT_VF_ARRAY = 1                                                    ## Default V-F array
    LIVE_VF_ARRAY = 2                                                       ## Live V-F array

class zes_vf_array_type_t(c_int):
    def __str__(self):
        return str(zes_vf_array_type_v(self.value))


###############################################################################
## @brief Overclock properties
## 
## @details
##     - Information on the overclock domain type and all the contols that are
##       part of the domain.
class zes_overclock_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("domainType", zes_overclock_domain_t),                         ## [out] The hardware block that this overclock domain controls (GPU,
                                                                        ## VRAM, ...)
        ("AvailableControls", c_ulong),                                 ## [out] Returns the overclock controls that are supported (a bit for
                                                                        ## each of enum ::zes_overclock_control_t). If no bits are set, the
                                                                        ## domain doesn't support overclocking.
        ("VFProgramType", zes_vf_program_type_t),                       ## [out] Type of V-F curve programming that is permitted:.
        ("NumberOfVFPoints", c_ulong)                                   ## [out] Number of VF points that can be programmed - max_num_points
    ]

###############################################################################
## @brief Overclock Control properties
## 
## @details
##     - Provides all the control capabilities supported by the device for the
##       overclock domain.
class zes_control_property_t(Structure):
    _fields_ = [
        ("MinValue", c_double),                                         ## [out]  This provides information about the limits of the control value
                                                                        ## so that the driver can calculate the set of valid values.
        ("MaxValue", c_double),                                         ## [out]  This provides information about the limits of the control value
                                                                        ## so that the driver can calculate the set of valid values.
        ("StepValue", c_double),                                        ## [out]  This provides information about the limits of the control value
                                                                        ## so that the driver can calculate the set of valid values.
        ("RefValue", c_double),                                         ## [out] The reference value provides the anchor point, UIs can combine
                                                                        ## this with the user offset request to show the anticipated improvement.
        ("DefaultValue", c_double)                                      ## [out] The shipped out-of-box position of this control. Driver can
                                                                        ## request this value at any time to return to the out-of-box behavior.
    ]

###############################################################################
## @brief Overclock VF properties
## 
## @details
##     - Provides all the VF capabilities supported by the device for the
##       overclock domain.
class zes_vf_property_t(Structure):
    _fields_ = [
        ("MinFreq", c_double),                                          ## [out] Read the minimum frequency that can be be programmed in the
                                                                        ## custom V-F point..
        ("MaxFreq", c_double),                                          ## [out] Read the maximum frequency that can be be programmed in the
                                                                        ## custom V-F point..
        ("StepFreq", c_double),                                         ## [out] Read the frequency step that can be be programmed in the custom
                                                                        ## V-F point..
        ("MinVolt", c_double),                                          ## [out] Read the minimum voltage that can be be programmed in the custom
                                                                        ## V-F point..
        ("MaxVolt", c_double),                                          ## [out] Read the maximum voltage that can be be programmed in the custom
                                                                        ## V-F point..
        ("StepVolt", c_double)                                          ## [out] Read the voltage step that can be be programmed in the custom
                                                                        ## V-F point.
    ]

###############################################################################
## @brief Diagnostic results
class zes_diag_result_v(IntEnum):
    NO_ERRORS = 0                                                           ## Diagnostic completed without finding errors to repair
    ABORT = 1                                                               ## Diagnostic had problems running tests
    FAIL_CANT_REPAIR = 2                                                    ## Diagnostic had problems setting up repairs
    REBOOT_FOR_REPAIR = 3                                                   ## Diagnostics found errors, setup for repair and reboot is required to
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
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("onSubdevice", ze_bool_t),                                     ## [out] True if the resource is located on a sub-device; false means
                                                                        ## that the resource is on the device of the calling Sysman handle
        ("subdeviceId", c_ulong),                                       ## [out] If onSubdevice is true, this gives the ID of the sub-device
        ("name", c_char * ZES_STRING_PROPERTY_SIZE),                    ## [out] Name of the diagnostics test suite
        ("haveTests", ze_bool_t)                                        ## [out] Indicates if this test suite has individual tests which can be
                                                                        ## run separately (use the function ::zesDiagnosticsGetTests() to get the
                                                                        ## list of these tests)
    ]

###############################################################################
## @brief ECC State
class zes_device_ecc_state_v(IntEnum):
    UNAVAILABLE = 0                                                         ## None
    ENABLED = 1                                                             ## ECC enabled.
    DISABLED = 2                                                            ## ECC disabled.

class zes_device_ecc_state_t(c_int):
    def __str__(self):
        return str(zes_device_ecc_state_v(self.value))


###############################################################################
## @brief State Change Requirements
class zes_device_action_v(IntEnum):
    NONE = 0                                                                ## No action.
    WARM_CARD_RESET = 1                                                     ## Warm reset of the card.
    COLD_CARD_RESET = 2                                                     ## Cold reset of the card.
    COLD_SYSTEM_REBOOT = 3                                                  ## Cold reboot of the system.

class zes_device_action_t(c_int):
    def __str__(self):
        return str(zes_device_action_v(self.value))


###############################################################################
## @brief ECC State Descriptor
class zes_device_ecc_desc_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("state", zes_device_ecc_state_t)                               ## [out] ECC state
    ]

###############################################################################
## @brief ECC State
class zes_device_ecc_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("currentState", zes_device_ecc_state_t),                       ## [out] Current ECC state
        ("pendingState", zes_device_ecc_state_t),                       ## [out] Pending ECC state
        ("pendingAction", zes_device_action_t)                          ## [out] Pending action
    ]

###############################################################################
## @brief Accelerator engine groups
class zes_engine_group_v(IntEnum):
    ALL = 0                                                                 ## Access information about all engines combined.
    COMPUTE_ALL = 1                                                         ## Access information about all compute engines combined. Compute engines
                                                                            ## can only process compute kernels (no 3D content).
    MEDIA_ALL = 2                                                           ## Access information about all media engines combined.
    COPY_ALL = 3                                                            ## Access information about all copy (blitter) engines combined.
    COMPUTE_SINGLE = 4                                                      ## Access information about a single compute engine - this is an engine
                                                                            ## that can process compute kernels. Note that single engines may share
                                                                            ## the same underlying accelerator resources as other engines so activity
                                                                            ## of such an engine may not be indicative of the underlying resource
                                                                            ## utilization - use ::ZES_ENGINE_GROUP_3D_RENDER_COMPUTE_ALL for that.
    RENDER_SINGLE = 5                                                       ## Access information about a single render engine - this is an engine
                                                                            ## that can process both 3D content and compute kernels. Note that single
                                                                            ## engines may share the same underlying accelerator resources as other
                                                                            ## engines so activity of such an engine may not be indicative of the
                                                                            ## underlying resource utilization - use
                                                                            ## ::ZES_ENGINE_GROUP_3D_RENDER_COMPUTE_ALL for that.
    MEDIA_DECODE_SINGLE = 6                                                 ## [DEPRECATED] No longer supported.
    MEDIA_ENCODE_SINGLE = 7                                                 ## [DEPRECATED] No longer supported.
    COPY_SINGLE = 8                                                         ## Access information about a single media encode engine. Note that
                                                                            ## single engines may share the same underlying accelerator resources as
                                                                            ## other engines so activity of such an engine may not be indicative of
                                                                            ## the underlying resource utilization - use ::ZES_ENGINE_GROUP_COPY_ALL
                                                                            ## for that.
    MEDIA_ENHANCEMENT_SINGLE = 9                                            ## Access information about a single media enhancement engine. Note that
                                                                            ## single engines may share the same underlying accelerator resources as
                                                                            ## other engines so activity of such an engine may not be indicative of
                                                                            ## the underlying resource utilization - use ::ZES_ENGINE_GROUP_MEDIA_ALL
                                                                            ## for that.
    _3D_SINGLE = 10                                                         ## [DEPRECATED] No longer supported.
    _3D_RENDER_COMPUTE_ALL = 11                                             ## [DEPRECATED] No longer supported.
    RENDER_ALL = 12                                                         ## Access information about all render engines combined. Render engines
                                                                            ## are those than process both 3D content and compute kernels.
    _3D_ALL = 13                                                            ## [DEPRECATED] No longer supported.
    MEDIA_CODEC_SINGLE = 14                                                 ## Access information about a single media engine. Note that single
                                                                            ## engines may share the same underlying accelerator resources as other
                                                                            ## engines so activity of such an engine may not be indicative of the
                                                                            ## underlying resource utilization - use ::ZES_ENGINE_GROUP_MEDIA_ALL for
                                                                            ## that.

class zes_engine_group_t(c_int):
    def __str__(self):
        return str(zes_engine_group_v(self.value))


###############################################################################
## @brief Engine group properties
class zes_engine_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
##     - The `activeTime` time units are implementation-specific since the
##       value is only intended to be used for calculating utilization
##       percentage.
##     - The `timestamp` should only be used to calculate delta between
##       snapshots of this structure.
##     - The application should never take the delta of `timestamp` with the
##       timestamp from a different structure since they are not guaranteed to
##       have the same base.
##     - When taking the delta, the difference between `timestamp` samples
##       could be `0`, if the frequency of sampling the snapshots is higher
##       than the frequency of the timestamp update.
##     - The absolute value of `timestamp` is only valid during within the
##       application and may be different on the next execution.
class zes_engine_stats_t(Structure):
    _fields_ = [
        ("activeTime", c_ulonglong),                                    ## [out] Monotonic counter where the resource is actively running
                                                                        ## workloads.
        ("timestamp", c_ulonglong)                                      ## [out] Monotonic counter when activeTime counter was sampled.
    ]

###############################################################################
## @brief Event types
class zes_event_type_flags_v(IntEnum):
    DEVICE_DETACH = ZE_BIT(0)                                               ## Event is triggered when the device is no longer available (due to a
                                                                            ## reset or being disabled).
    DEVICE_ATTACH = ZE_BIT(1)                                               ## Event is triggered after the device is available again.
    DEVICE_SLEEP_STATE_ENTER = ZE_BIT(2)                                    ## Event is triggered when the driver is about to put the device into a
                                                                            ## deep sleep state
    DEVICE_SLEEP_STATE_EXIT = ZE_BIT(3)                                     ## Event is triggered when the driver is waking the device up from a deep
                                                                            ## sleep state
    FREQ_THROTTLED = ZE_BIT(4)                                              ## Event is triggered when the frequency starts being throttled
    ENERGY_THRESHOLD_CROSSED = ZE_BIT(5)                                    ## Event is triggered when the energy consumption threshold is reached
                                                                            ## (use ::zesPowerSetEnergyThreshold() to configure).
    TEMP_CRITICAL = ZE_BIT(6)                                               ## Event is triggered when the critical temperature is reached (use
                                                                            ## ::zesTemperatureSetConfig() to configure - disabled by default).
    TEMP_THRESHOLD1 = ZE_BIT(7)                                             ## Event is triggered when the temperature crosses threshold 1 (use
                                                                            ## ::zesTemperatureSetConfig() to configure - disabled by default).
    TEMP_THRESHOLD2 = ZE_BIT(8)                                             ## Event is triggered when the temperature crosses threshold 2 (use
                                                                            ## ::zesTemperatureSetConfig() to configure - disabled by default).
    MEM_HEALTH = ZE_BIT(9)                                                  ## Event is triggered when the health of device memory changes.
    FABRIC_PORT_HEALTH = ZE_BIT(10)                                         ## Event is triggered when the health of fabric ports change.
    PCI_LINK_HEALTH = ZE_BIT(11)                                            ## Event is triggered when the health of the PCI link changes.
    RAS_CORRECTABLE_ERRORS = ZE_BIT(12)                                     ## Event is triggered when accelerator RAS correctable errors cross
                                                                            ## thresholds (use ::zesRasSetConfig() to configure - disabled by
                                                                            ## default).
    RAS_UNCORRECTABLE_ERRORS = ZE_BIT(13)                                   ## Event is triggered when accelerator RAS uncorrectable errors cross
                                                                            ## thresholds (use ::zesRasSetConfig() to configure - disabled by
                                                                            ## default).
    DEVICE_RESET_REQUIRED = ZE_BIT(14)                                      ## Event is triggered when the device needs to be reset (use
                                                                            ## ::zesDeviceGetState() to determine the reasons for the reset).
    SURVIVABILITY_MODE_DETECTED = ZE_BIT(15)                                ## Event is triggered when graphics driver encounter an error condition.

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
    UNKNOWN = 0                                                             ## The port status cannot be determined
    HEALTHY = 1                                                             ## The port is up and operating as expected
    DEGRADED = 2                                                            ## The port is up but has quality and/or speed degradation
    FAILED = 3                                                              ## Port connection instabilities are preventing workloads making forward
                                                                            ## progress
    DISABLED = 4                                                            ## The port is configured down

class zes_fabric_port_status_t(c_int):
    def __str__(self):
        return str(zes_fabric_port_status_v(self.value))


###############################################################################
## @brief Fabric port quality degradation reasons
class zes_fabric_port_qual_issue_flags_v(IntEnum):
    LINK_ERRORS = ZE_BIT(0)                                                 ## Excessive link errors are occurring
    SPEED = ZE_BIT(1)                                                       ## There is a degradation in the bitrate and/or width of the link

class zes_fabric_port_qual_issue_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Fabric port failure reasons
class zes_fabric_port_failure_flags_v(IntEnum):
    FAILED = ZE_BIT(0)                                                      ## A previously operating link has failed. Hardware will automatically
                                                                            ## retrain this port. This state will persist until either the physical
                                                                            ## connection is removed or the link trains successfully.
    TRAINING_TIMEOUT = ZE_BIT(1)                                            ## A connection has not been established within an expected time.
                                                                            ## Hardware will continue to attempt port training. This status will
                                                                            ## persist until either the physical connection is removed or the link
                                                                            ## successfully trains.
    FLAPPING = ZE_BIT(2)                                                    ## Port has excessively trained and then transitioned down for some
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
##       all fabric ports and match the `remotePortId` member of
##       ::zes_fabric_port_state_t to the `portId` member of
##       ::zes_fabric_port_properties_t.
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
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
        ("desc", c_char * ZES_MAX_FABRIC_LINK_TYPE_SIZE)                ## [out] Description of link technology. Will be set to the string
                                                                        ## "unkown" if this cannot be determined for this link.
    ]

###############################################################################
## @brief Fabric port configuration
class zes_fabric_port_config_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("enabled", ze_bool_t),                                         ## [in,out] Port is configured up/down
        ("beaconing", ze_bool_t)                                        ## [in,out] Beaconing is configured on/off
    ]

###############################################################################
## @brief Fabric port state
class zes_fabric_port_state_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
## @brief Fabric Port Error Counters
class zes_fabric_port_error_counters_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("linkFailureCount", c_ulonglong),                              ## [out] Link Failure Error Count reported per port
        ("fwCommErrorCount", c_ulonglong),                              ## [out] Firmware Communication Error Count reported per device
        ("fwErrorCount", c_ulonglong),                                  ## [out] Firmware reported Error Count reported per device
        ("linkDegradeCount", c_ulonglong)                               ## [out] Link Degrade Error Count reported per port
    ]

###############################################################################
## @brief Fan resource speed mode
class zes_fan_speed_mode_v(IntEnum):
    DEFAULT = 0                                                             ## The fan speed is operating using the hardware default settings
    FIXED = 1                                                               ## The fan speed is currently set to a fixed value
    TABLE = 2                                                               ## The fan speed is currently controlled dynamically by hardware based on
                                                                            ## a temp/speed table

class zes_fan_speed_mode_t(c_int):
    def __str__(self):
        return str(zes_fan_speed_mode_v(self.value))


###############################################################################
## @brief Fan speed units
class zes_fan_speed_units_v(IntEnum):
    RPM = 0                                                                 ## The fan speed is in units of revolutions per minute (rpm)
    PERCENT = 1                                                             ## The fan speed is a percentage of the maximum speed of the fan

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
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("mode", zes_fan_speed_mode_t),                                 ## [in,out] The fan speed mode (fixed, temp-speed table)
        ("speedFixed", zes_fan_speed_t),                                ## [in,out] The current fixed fan speed setting
        ("speedTable", zes_fan_speed_table_t)                           ## [out] A table containing temperature/speed pairs
    ]

###############################################################################
## @brief Firmware properties
class zes_firmware_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
    GPU = 0                                                                 ## GPU Core Domain.
    MEMORY = 1                                                              ## Local Memory Domain.
    MEDIA = 2                                                               ## GPU Media Domain.

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
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
## @brief Frequency range between which the hardware can operate.
## 
## @details
##     - When setting limits, they will be clamped to the hardware limits.
##     - When setting limits, ensure that the max frequency is greater than or
##       equal to the min frequency specified.
##     - When setting limits to return to factory settings, specify -1 for both
##       the min and max limit.
class zes_freq_range_t(Structure):
    _fields_ = [
        ("min", c_double),                                              ## [in,out] The min frequency in MHz below which hardware frequency
                                                                        ## management will not request frequencies. On input, setting to 0 will
                                                                        ## permit the frequency to go down to the hardware minimum while setting
                                                                        ## to -1 will return the min frequency limit to the factory value (can be
                                                                        ## larger than the hardware min). On output, a negative value indicates
                                                                        ## that no external minimum frequency limit is in effect.
        ("max", c_double)                                               ## [in,out] The max frequency in MHz above which hardware frequency
                                                                        ## management will not request frequencies. On input, setting to 0 or a
                                                                        ## very big number will permit the frequency to go all the way up to the
                                                                        ## hardware maximum while setting to -1 will return the max frequency to
                                                                        ## the factory value (which can be less than the hardware max). On
                                                                        ## output, a negative number indicates that no external maximum frequency
                                                                        ## limit is in effect.
    ]

###############################################################################
## @brief Frequency throttle reasons
class zes_freq_throttle_reason_flags_v(IntEnum):
    AVE_PWR_CAP = ZE_BIT(0)                                                 ## frequency throttled due to average power excursion (PL1)
    BURST_PWR_CAP = ZE_BIT(1)                                               ## frequency throttled due to burst power excursion (PL2)
    CURRENT_LIMIT = ZE_BIT(2)                                               ## frequency throttled due to current excursion (PL4)
    THERMAL_LIMIT = ZE_BIT(3)                                               ## frequency throttled due to thermal excursion (T > TjMax)
    PSU_ALERT = ZE_BIT(4)                                                   ## frequency throttled due to power supply assertion
    SW_RANGE = ZE_BIT(5)                                                    ## frequency throttled due to software supplied frequency range
    HW_RANGE = ZE_BIT(6)                                                    ## frequency throttled due to a sub block that has a lower frequency
                                                                            ## range when it receives clocks

class zes_freq_throttle_reason_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Frequency state
class zes_freq_state_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
## 
## @details
##     - [DEPRECATED] No longer supported.
class zes_oc_mode_v(IntEnum):
    OFF = 0                                                                 ## Overclocking if off - hardware is running using factory default
                                                                            ## voltages/frequencies.
    OVERRIDE = 1                                                            ## Overclock override mode - In this mode, a fixed user-supplied voltage
                                                                            ## is applied independent of the frequency request. The maximum permitted
                                                                            ## frequency can also be increased. This mode disables INTERPOLATIVE and
                                                                            ## FIXED modes.
    INTERPOLATIVE = 2                                                       ## Overclock interpolative mode - In this mode, the voltage/frequency
                                                                            ## curve can be extended with a new voltage/frequency point that will be
                                                                            ## interpolated. The existing voltage/frequency points can also be offset
                                                                            ## (up or down) by a fixed voltage. This mode disables FIXED and OVERRIDE
                                                                            ## modes.
    FIXED = 3                                                               ## Overclocking fixed Mode - In this mode, hardware will disable most
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
##     - [DEPRECATED] No longer supported.
class zes_oc_capabilities_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("isOn", ze_bool_t),                                            ## [out] Indicates if the LED is on or off
        ("color", zes_led_color_t)                                      ## [out] Color of the LED
    ]

###############################################################################
## @brief Memory module types
class zes_mem_type_v(IntEnum):
    HBM = 0                                                                 ## HBM memory
    DDR = 1                                                                 ## DDR memory
    DDR3 = 2                                                                ## DDR3 memory
    DDR4 = 3                                                                ## DDR4 memory
    DDR5 = 4                                                                ## DDR5 memory
    LPDDR = 5                                                               ## LPDDR memory
    LPDDR3 = 6                                                              ## LPDDR3 memory
    LPDDR4 = 7                                                              ## LPDDR4 memory
    LPDDR5 = 8                                                              ## LPDDR5 memory
    SRAM = 9                                                                ## SRAM memory
    L1 = 10                                                                 ## L1 cache
    L3 = 11                                                                 ## L3 cache
    GRF = 12                                                                ## Execution unit register file
    SLM = 13                                                                ## Execution unit shared local memory
    GDDR4 = 14                                                              ## GDDR4 memory
    GDDR5 = 15                                                              ## GDDR5 memory
    GDDR5X = 16                                                             ## GDDR5X memory
    GDDR6 = 17                                                              ## GDDR6 memory
    GDDR6X = 18                                                             ## GDDR6X memory
    GDDR7 = 19                                                              ## GDDR7 memory

class zes_mem_type_t(c_int):
    def __str__(self):
        return str(zes_mem_type_v(self.value))


###############################################################################
## @brief Memory module location
class zes_mem_loc_v(IntEnum):
    SYSTEM = 0                                                              ## System memory
    DEVICE = 1                                                              ## On board local device memory

class zes_mem_loc_t(c_int):
    def __str__(self):
        return str(zes_mem_loc_v(self.value))


###############################################################################
## @brief Memory health
class zes_mem_health_v(IntEnum):
    UNKNOWN = 0                                                             ## The memory health cannot be determined.
    OK = 1                                                                  ## All memory channels are healthy.
    DEGRADED = 2                                                            ## Excessive correctable errors have been detected on one or more
                                                                            ## channels. Device should be reset.
    CRITICAL = 3                                                            ## Operating with reduced memory to cover banks with too many
                                                                            ## uncorrectable errors.
    REPLACE = 4                                                             ## Device should be replaced due to excessive uncorrectable errors.

class zes_mem_health_t(c_int):
    def __str__(self):
        return str(zes_mem_health_v(self.value))


###############################################################################
## @brief Memory properties
class zes_mem_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("health", zes_mem_health_t),                                   ## [out] Indicates the health of the memory
        ("free", c_ulonglong),                                          ## [out] The free memory in bytes
        ("size", c_ulonglong)                                           ## [out] The total allocatable memory in bytes (can be less than the
                                                                        ## `physicalSize` member of ::zes_mem_properties_t)
    ]

###############################################################################
## @brief Memory bandwidth
## 
## @details
##     - Percent bandwidth is calculated by taking two snapshots (s1, s2) and
##       using the equation: %bw = 10^6 * ((s2.readCounter - s1.readCounter) +
##       (s2.writeCounter - s1.writeCounter)) / (s2.maxBandwidth *
##       (s2.timestamp - s1.timestamp))
##     - Counter can roll over and rollover needs to be handled by comparing
##       the current read against the previous read
##     - Counter is a 32 byte transaction count, which means the calculated
##       delta (delta = current_value - previous_value or delta = 2^32 -
##       previous_value + current_value in case of rollover) needs to be
##       multiplied by 32 to get delta between samples in actual byte count
class zes_mem_bandwidth_t(Structure):
    _fields_ = [
        ("readCounter", c_ulonglong),                                   ## [out] Total bytes read from memory
        ("writeCounter", c_ulonglong),                                  ## [out] Total bytes written to memory
        ("maxBandwidth", c_ulonglong),                                  ## [out] Current maximum bandwidth in units of bytes/sec
        ("timestamp", c_ulonglong)                                      ## [out] The timestamp in microseconds when these measurements were sampled.
                                                                        ## This timestamp should only be used to calculate delta time between
                                                                        ## snapshots of this structure.
                                                                        ## Never take the delta of this timestamp with the timestamp from a
                                                                        ## different structure since they are not guaranteed to have the same base.
                                                                        ## The absolute value of the timestamp is only valid during within the
                                                                        ## application and may be different on the next execution.
    ]

###############################################################################
## @brief Extension properties for Memory bandwidth
## 
## @details
##     - Number of counter bits
##     - [DEPRECATED] No longer supported.
class zes_mem_ext_bandwidth_t(Structure):
    _fields_ = [
        ("memoryTimestampValidBits", c_ulong)                           ## [out] Returns the number of valid bits in the timestamp values
    ]

###############################################################################
## @brief Static information about a Performance Factor domain
class zes_perf_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("onSubdevice", ze_bool_t),                                     ## [out] True if this Performance Factor affects accelerators located on
                                                                        ## a sub-device
        ("subdeviceId", c_ulong),                                       ## [out] If onSubdevice is true, this gives the ID of the sub-device
        ("engines", zes_engine_type_flags_t)                            ## [out] Bitfield of accelerator engine types that are affected by this
                                                                        ## Performance Factor.
    ]

###############################################################################
## @brief Power Domain
class zes_power_domain_v(IntEnum):
    UNKNOWN = 0                                                             ## The PUnit power domain level cannot be determined.
    CARD = 1                                                                ## The PUnit power domain is a card-level power domain.
    PACKAGE = 2                                                             ## The PUnit power domain is a package-level power domain.
    STACK = 3                                                               ## The PUnit power domain is a stack-level power domain.
    MEMORY = 4                                                              ## The PUnit power domain is a memory-level power domain.
    GPU = 5                                                                 ## The PUnit power domain is a GPU-level power domain.

class zes_power_domain_t(c_int):
    def __str__(self):
        return str(zes_power_domain_v(self.value))


###############################################################################
## @brief Power Level Type
class zes_power_level_v(IntEnum):
    UNKNOWN = 0                                                             ## The PUnit power monitoring duration cannot be determined.
    SUSTAINED = 1                                                           ## The PUnit determines effective power draw by computing a moving
                                                                            ## average of the actual power draw over a time interval (longer than
                                                                            ## BURST).
    BURST = 2                                                               ## The PUnit determines effective power draw by computing a moving
                                                                            ## average of the actual power draw over a time interval (longer than
                                                                            ## PEAK).
    PEAK = 3                                                                ## The PUnit determines effective power draw by computing a moving
                                                                            ## average of the actual power draw over a very short time interval.
    INSTANTANEOUS = 4                                                       ## The PUnit predicts effective power draw using the current device
                                                                            ## configuration (frequency, voltage, etc...) & throttles proactively to
                                                                            ## stay within the specified limit.

class zes_power_level_t(c_int):
    def __str__(self):
        return str(zes_power_level_v(self.value))


###############################################################################
## @brief Power Source Type
class zes_power_source_v(IntEnum):
    ANY = 0                                                                 ## Limit active no matter whether the power source is mains powered or
                                                                            ## battery powered.
    MAINS = 1                                                               ## Limit active only when the device is mains powered.
    BATTERY = 2                                                             ## Limit active only when the device is battery powered.

class zes_power_source_t(c_int):
    def __str__(self):
        return str(zes_power_source_v(self.value))


###############################################################################
## @brief Limit Unit
class zes_limit_unit_v(IntEnum):
    UNKNOWN = 0                                                             ## The PUnit power monitoring unit cannot be determined.
    CURRENT = 1                                                             ## The limit is specified in milliamperes of current drawn.
    POWER = 2                                                               ## The limit is specified in milliwatts of power generated.

class zes_limit_unit_t(c_int):
    def __str__(self):
        return str(zes_limit_unit_v(self.value))


###############################################################################
## @brief Properties related to device power settings
class zes_power_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("onSubdevice", ze_bool_t),                                     ## [out] True if this resource is located on a sub-device; false means
                                                                        ## that the resource is on the device of the calling Sysman handle
        ("subdeviceId", c_ulong),                                       ## [out] If onSubdevice is true, this gives the ID of the sub-device
        ("canControl", ze_bool_t),                                      ## [out] Software can change the power limits of this domain assuming the
                                                                        ## user has permissions.
        ("isEnergyThresholdSupported", ze_bool_t),                      ## [out] Indicates if this power domain supports the energy threshold
                                                                        ## event (::ZES_EVENT_TYPE_FLAG_ENERGY_THRESHOLD_CROSSED).
        ("defaultLimit", c_int32_t),                                    ## [out] (Deprecated) The factory default TDP power limit of the part in
                                                                        ## milliwatts. A value of -1 means that this is not known.
        ("minLimit", c_int32_t),                                        ## [out] (Deprecated) The minimum power limit in milliwatts that can be
                                                                        ## requested. A value of -1 means that this is not known.
        ("maxLimit", c_int32_t)                                         ## [out] (Deprecated) The maximum power limit in milliwatts that can be
                                                                        ## requested. A value of -1 means that this is not known.
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
##     - [DEPRECATED] No longer supported.
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
##     - [DEPRECATED] No longer supported.
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
##     - [DEPRECATED] No longer supported.
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
    UNKNOWN = 0                                                             ## The status of the power supply voltage controllers cannot be
                                                                            ## determined
    NORMAL = 1                                                              ## No unusual voltages have been detected
    OVER = 2                                                                ## Over-voltage has occurred
    UNDER = 3                                                               ## Under-voltage has occurred

class zes_psu_voltage_status_t(c_int):
    def __str__(self):
        return str(zes_psu_voltage_status_v(self.value))


###############################################################################
## @brief Static properties of the power supply
class zes_psu_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
    CORRECTABLE = 0                                                         ## Errors were corrected by hardware
    UNCORRECTABLE = 1                                                       ## Error were not corrected

class zes_ras_error_type_t(c_int):
    def __str__(self):
        return str(zes_ras_error_type_v(self.value))


###############################################################################
## @brief RAS error categories
class zes_ras_error_cat_v(IntEnum):
    RESET = 0                                                               ## The number of accelerator engine resets attempted by the driver
    PROGRAMMING_ERRORS = 1                                                  ## The number of hardware exceptions generated by the way workloads have
                                                                            ## programmed the hardware
    DRIVER_ERRORS = 2                                                       ## The number of low level driver communication errors have occurred
    COMPUTE_ERRORS = 3                                                      ## The number of errors that have occurred in the compute accelerator
                                                                            ## hardware
    NON_COMPUTE_ERRORS = 4                                                  ## The number of errors that have occurred in the fixed-function
                                                                            ## accelerator hardware
    CACHE_ERRORS = 5                                                        ## The number of errors that have occurred in caches (L1/L3/register
                                                                            ## file/shared local memory/sampler)
    DISPLAY_ERRORS = 6                                                      ## The number of errors that have occurred in the display

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
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
    TIMEOUT = 0                                                             ## Multiple applications or contexts are submitting work to the hardware.
                                                                            ## When higher priority work arrives, the scheduler attempts to pause the
                                                                            ## current executing work within some timeout interval, then submits the
                                                                            ## other work.
    TIMESLICE = 1                                                           ## The scheduler attempts to fairly timeslice hardware execution time
                                                                            ## between multiple contexts submitting work to the hardware
                                                                            ## concurrently.
    EXCLUSIVE = 2                                                           ## Any application or context can run indefinitely on the hardware
                                                                            ## without being preempted or terminated. All pending work for other
                                                                            ## contexts must wait until the running context completes with no further
                                                                            ## submitted work.
    COMPUTE_UNIT_DEBUG = 3                                                  ## [DEPRECATED] No longer supported.

class zes_sched_mode_t(c_int):
    def __str__(self):
        return str(zes_sched_mode_v(self.value))


###############################################################################
## @brief Properties related to scheduler component
class zes_sched_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
    GLOBAL = 0                                                              ## Control the overall standby policy of the device/sub-device

class zes_standby_type_t(c_int):
    def __str__(self):
        return str(zes_standby_type_v(self.value))


###############################################################################
## @brief Standby hardware component properties
class zes_standby_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("type", zes_standby_type_t),                                   ## [out] Which standby hardware component this controls
        ("onSubdevice", ze_bool_t),                                     ## [out] True if the resource is located on a sub-device; false means
                                                                        ## that the resource is on the device of the calling Sysman handle
        ("subdeviceId", c_ulong)                                        ## [out] If onSubdevice is true, this gives the ID of the sub-device
    ]

###############################################################################
## @brief Standby promotion modes
class zes_standby_promo_mode_v(IntEnum):
    DEFAULT = 0                                                             ## Best compromise between performance and energy savings.
    NEVER = 1                                                               ## The device/component will never shutdown. This can improve performance
                                                                            ## but uses more energy.

class zes_standby_promo_mode_t(c_int):
    def __str__(self):
        return str(zes_standby_promo_mode_v(self.value))


###############################################################################
## @brief Temperature sensors
class zes_temp_sensors_v(IntEnum):
    GLOBAL = 0                                                              ## The maximum temperature across all device sensors
    GPU = 1                                                                 ## The maximum temperature across all sensors in the GPU
    MEMORY = 2                                                              ## The maximum temperature across all sensors in the local memory
    GLOBAL_MIN = 3                                                          ## The minimum temperature across all device sensors
    GPU_MIN = 4                                                             ## The minimum temperature across all sensors in the GPU
    MEMORY_MIN = 5                                                          ## The minimum temperature across all sensors in the local device memory
    GPU_BOARD = 6                                                           ## The maximum temperature across all sensors in the GPU Board
    GPU_BOARD_MIN = 7                                                       ## The minimum temperature across all sensors in the GPU Board
    VOLTAGE_REGULATOR = 8                                                   ## The maximum temperature across all sensors in the Voltage Regulator

class zes_temp_sensors_t(c_int):
    def __str__(self):
        return str(zes_temp_sensors_v(self.value))


###############################################################################
## @brief Temperature sensor properties
class zes_temp_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
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
## @brief Power Limits Extension Name
ZES_POWER_LIMITS_EXT_NAME = "ZES_extension_power_limits"

###############################################################################
## @brief Power Limits Extension Version(s)
class zes_power_limits_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class zes_power_limits_ext_version_t(c_int):
    def __str__(self):
        return str(zes_power_limits_ext_version_v(self.value))


###############################################################################
## @brief Device power/current limit descriptor.
class zes_power_limit_ext_desc_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("level", zes_power_level_t),                                   ## [in,out] duration type over which the power draw is measured, i.e.
                                                                        ## sustained, burst, peak, or critical.
        ("source", zes_power_source_t),                                 ## [out] source of power used by the system, i.e. AC or DC.
        ("limitUnit", zes_limit_unit_t),                                ## [out] unit used for specifying limit, i.e. current units (milliamps)
                                                                        ## or power units (milliwatts).
        ("enabledStateLocked", ze_bool_t),                              ## [out] indicates if the power limit state (enabled/ignored) can be set
                                                                        ## (false) or is locked (true).
        ("enabled", ze_bool_t),                                         ## [in,out] indicates if the limit is enabled (true) or ignored (false).
                                                                        ## If enabledStateIsLocked is True, this value is ignored.
        ("intervalValueLocked", ze_bool_t),                             ## [out] indicates if the interval can be modified (false) or is fixed
                                                                        ## (true).
        ("interval", c_int32_t),                                        ## [in,out] power averaging window in milliseconds. If
                                                                        ## intervalValueLocked is true, this value is ignored.
        ("limitValueLocked", ze_bool_t),                                ## [out] indicates if the limit can be set (false) or if the limit is
                                                                        ## fixed (true).
        ("limit", c_int32_t)                                            ## [in,out] limit value. If limitValueLocked is true, this value is
                                                                        ## ignored. The value should be provided in the unit specified by
                                                                        ## limitUnit.
    ]

###############################################################################
## @brief Extension properties related to device power settings
## 
## @details
##     - This structure may be returned from ::zesPowerGetProperties via the
##       `pNext` member of ::zes_power_properties_t.
##     - This structure may also be returned from ::zesPowerGetProperties via
##       the `pNext` member of ::zes_power_ext_properties_t
##     - Used for determining the power domain level, i.e. card-level v/s
##       package-level v/s stack-level & the factory default power limits.
class zes_power_ext_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("domain", zes_power_domain_t),                                 ## [out] domain that the power limit belongs to.
        ("defaultLimit", POINTER(zes_power_limit_ext_desc_t))           ## [out] the factory default limit of the part.
    ]

###############################################################################
## @brief Engine Activity Extension Name
ZES_ENGINE_ACTIVITY_EXT_NAME = "ZES_extension_engine_activity"

###############################################################################
## @brief Engine Activity Extension Version(s)
class zes_engine_activity_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class zes_engine_activity_ext_version_t(c_int):
    def __str__(self):
        return str(zes_engine_activity_ext_version_v(self.value))


###############################################################################
## @brief Extension properties related to Engine Groups
## 
## @details
##     - This structure may be passed to ::zesEngineGetProperties by having the
##       pNext member of ::zes_engine_properties_t point at this struct.
##     - Used for SRIOV per Virtual Function device utilization by
##       ::zes_engine_group_t
class zes_engine_ext_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("countOfVirtualFunctionInstance", c_ulong)                     ## [out] Number of Virtual Function(VF) instances associated with engine
                                                                        ## to monitor the utilization of hardware across all Virtual Function
                                                                        ## from a Physical Function (PF) instance.
                                                                        ## These VF-by-VF views should provide engine group and individual engine
                                                                        ## level granularity.
                                                                        ## This count represents the number of VF instances that are actively
                                                                        ## using the resource represented by the engine handle.
    ]

###############################################################################
## @brief RAS Get State Extension Name
ZES_RAS_GET_STATE_EXP_NAME = "ZES_extension_ras_state"

###############################################################################
## @brief RAS Get State Extension Version(s)
class zes_ras_state_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class zes_ras_state_exp_version_t(c_int):
    def __str__(self):
        return str(zes_ras_state_exp_version_v(self.value))


###############################################################################
## @brief RAS error categories
class zes_ras_error_category_exp_v(IntEnum):
    RESET = 0                                                               ## The number of accelerator engine resets attempted by the driver
    PROGRAMMING_ERRORS = 1                                                  ## The number of hardware exceptions generated by the way workloads have
                                                                            ## programmed the hardware
    DRIVER_ERRORS = 2                                                       ## The number of low level driver communication errors have occurred
    COMPUTE_ERRORS = 3                                                      ## The number of errors that have occurred in the compute accelerator
                                                                            ## hardware
    NON_COMPUTE_ERRORS = 4                                                  ## The number of errors that have occurred in the fixed-function
                                                                            ## accelerator hardware
    CACHE_ERRORS = 5                                                        ## The number of errors that have occurred in caches (L1/L3/register
                                                                            ## file/shared local memory/sampler)
    DISPLAY_ERRORS = 6                                                      ## The number of errors that have occurred in the display
    MEMORY_ERRORS = 7                                                       ## The number of errors that have occurred in Memory
    SCALE_ERRORS = 8                                                        ## The number of errors that have occurred in Scale Fabric
    L3FABRIC_ERRORS = 9                                                     ## The number of errors that have occurred in L3 Fabric

class zes_ras_error_category_exp_t(c_int):
    def __str__(self):
        return str(zes_ras_error_category_exp_v(self.value))


###############################################################################
## @brief Extension structure for providing RAS error counters for different
##        error sets
class zes_ras_state_exp_t(Structure):
    _fields_ = [
        ("category", zes_ras_error_category_exp_t),                     ## [out] category for which error counter is provided.
        ("errorCounter", c_ulonglong)                                   ## [out] Current value of RAS counter for specific error category.
    ]

###############################################################################
## @brief Memory State Extension Name
ZES_MEM_PAGE_OFFLINE_STATE_EXP_NAME = "ZES_extension_mem_state"

###############################################################################
## @brief Memory State Extension Version(s)
class zes_mem_page_offline_state_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class zes_mem_page_offline_state_exp_version_t(c_int):
    def __str__(self):
        return str(zes_mem_page_offline_state_exp_version_v(self.value))


###############################################################################
## @brief Extension properties for Memory State
## 
## @details
##     - This structure may be returned from ::zesMemoryGetState via the
##       `pNext` member of ::zes_mem_state_t
##     - These additional parameters get Memory Page Offline Metrics
class zes_mem_page_offline_state_exp_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("memoryPageOffline", c_ulong),                                 ## [out] Returns the number of Memory Pages Offline
        ("maxMemoryPageOffline", c_ulong)                               ## [out] Returns the Allowed Memory Pages Offline
    ]

###############################################################################
## @brief Memory Bandwidth Counter Valid Bits Extension Name
ZES_MEMORY_BANDWIDTH_COUNTER_BITS_EXP_PROPERTIES_NAME = "ZES_extension_mem_bandwidth_counter_bits_properties"

###############################################################################
## @brief Memory Bandwidth Counter Valid Bits Extension Version(s)
class zes_mem_bandwidth_counter_bits_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class zes_mem_bandwidth_counter_bits_exp_version_t(c_int):
    def __str__(self):
        return str(zes_mem_bandwidth_counter_bits_exp_version_v(self.value))


###############################################################################
## @brief Extension properties for reporting valid bit count for memory
##        bandwidth counter value
## 
## @details
##     - Number of valid read and write counter bits of memory bandwidth
##     - This structure may be returned from ::zesMemoryGetProperties via the
##       `pNext` member of ::zes_mem_properties_t.
##     - Used for denoting number of valid bits in the counter value returned
##       in ::zes_mem_bandwidth_t.
class zes_mem_bandwidth_counter_bits_exp_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("validBitsCount", c_ulong)                                     ## [out] Returns the number of valid bits in the counter values
    ]

###############################################################################
## @brief Power Domain Properties Name
ZES_POWER_DOMAIN_PROPERTIES_EXP_NAME = "ZES_extension_power_domain_properties"

###############################################################################
## @brief Power Domain Properties Extension Version(s)
class zes_power_domain_properties_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class zes_power_domain_properties_exp_version_t(c_int):
    def __str__(self):
        return str(zes_power_domain_properties_exp_version_v(self.value))


###############################################################################
## @brief Extension structure for providing power domain information associated
##        with a power handle
## 
## @details
##     - This structure may be returned from ::zesPowerGetProperties via the
##       `pNext` member of ::zes_power_properties_t.
##     - Used for associating a power handle with a power domain.
class zes_power_domain_exp_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("powerDomain", zes_power_domain_t)                             ## [out] Power domain associated with the power handle.
    ]

###############################################################################
## @brief Firmware security version
ZES_FIRMWARE_SECURITY_VERSION_EXP_NAME = "ZES_experimental_firmware_security_version"

###############################################################################
## @brief Firmware security version Extension Version(s)
class zes_firmware_security_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class zes_firmware_security_exp_version_t(c_int):
    def __str__(self):
        return str(zes_firmware_security_exp_version_v(self.value))


###############################################################################
## @brief Sysman Device Mapping Extension Name
ZES_SYSMAN_DEVICE_MAPPING_EXP_NAME = "ZES_experimental_sysman_device_mapping"

###############################################################################
## @brief Sysman Device Mapping Extension Version(s)
class zes_sysman_device_mapping_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class zes_sysman_device_mapping_exp_version_t(c_int):
    def __str__(self):
        return str(zes_sysman_device_mapping_exp_version_v(self.value))


###############################################################################
## @brief Sub Device Properties
class zes_subdevice_exp_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("subdeviceId", c_ulong),                                       ## [out] this gives the ID of the sub device
        ("uuid", zes_uuid_t)                                            ## [out] universal unique identifier of the sub device.
    ]

###############################################################################
## @brief Virtual Function Management Extension Name
ZES_VIRTUAL_FUNCTION_MANAGEMENT_EXP_NAME = "ZES_experimental_virtual_function_management"

###############################################################################
## @brief Virtual Function Management Extension Version(s)
class zes_vf_management_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0 (deprecated)
    _1_1 = ZE_MAKE_VERSION( 1, 1 )                                          ## version 1.1 (deprecated)
    _1_2 = ZE_MAKE_VERSION( 1, 2 )                                          ## version 1.2
    CURRENT = ZE_MAKE_VERSION( 1, 2 )                                       ## latest known version

class zes_vf_management_exp_version_t(c_int):
    def __str__(self):
        return str(zes_vf_management_exp_version_v(self.value))


###############################################################################
## @brief Virtual function memory types (deprecated)
class zes_vf_info_mem_type_exp_flags_v(IntEnum):
    MEM_TYPE_SYSTEM = ZE_BIT(0)                                             ## System memory
    MEM_TYPE_DEVICE = ZE_BIT(1)                                             ## Device local memory

class zes_vf_info_mem_type_exp_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Virtual function utilization flag bit fields (deprecated)
class zes_vf_info_util_exp_flags_v(IntEnum):
    INFO_NONE = ZE_BIT(0)                                                   ## No info associated with virtual function
    INFO_MEM_CPU = ZE_BIT(1)                                                ## System memory utilization associated with virtual function
    INFO_MEM_GPU = ZE_BIT(2)                                                ## Device memory utilization associated with virtual function
    INFO_ENGINE = ZE_BIT(3)                                                 ## Engine utilization associated with virtual function

class zes_vf_info_util_exp_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Virtual function management properties (deprecated)
class zes_vf_exp_properties_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("address", zes_pci_address_t),                                 ## [out] Virtual function BDF address
        ("uuid", zes_uuid_t),                                           ## [out] universal unique identifier of the device
        ("flags", zes_vf_info_util_exp_flags_t)                         ## [out] utilization flags available. May be 0 or a valid combination of
                                                                        ## ::zes_vf_info_util_exp_flag_t.
    ]

###############################################################################
## @brief Provides memory utilization values for a virtual function (deprecated)
class zes_vf_util_mem_exp_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("memTypeFlags", zes_vf_info_mem_type_exp_flags_t),             ## [out] Memory type flags.
        ("free", c_ulonglong),                                          ## [out] Free memory size in bytes.
        ("size", c_ulonglong),                                          ## [out] Total allocatable memory in bytes.
        ("timestamp", c_ulonglong)                                      ## [out] Wall clock time from VF when value was sampled.
    ]

###############################################################################
## @brief Provides engine utilization values for a virtual function (deprecated)
class zes_vf_util_engine_exp_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("type", zes_engine_group_t),                                   ## [out] The engine group.
        ("activeCounterValue", c_ulonglong),                            ## [out] Represents active counter.
        ("samplingCounterValue", c_ulonglong),                          ## [out] Represents counter value when activeCounterValue was sampled.
        ("timestamp", c_ulonglong)                                      ## [out] Wall clock time when the activeCounterValue was sampled.
    ]

###############################################################################
## @brief Virtual function management capabilities
class zes_vf_exp_capabilities_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("address", zes_pci_address_t),                                 ## [out] Virtual function BDF address
        ("vfDeviceMemSize", c_ulong),                                   ## [out] Virtual function memory size in bytes
        ("vfID", c_ulong)                                               ## [out] Virtual Function ID
    ]

###############################################################################
## @brief Provides memory utilization values for a virtual function
class zes_vf_util_mem_exp2_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("vfMemLocation", zes_mem_loc_t),                               ## [out] Location of this memory (system, device)
        ("vfMemUtilized", c_ulonglong)                                  ## [out] Free memory size in bytes.
    ]

###############################################################################
## @brief Provides engine utilization values for a virtual function
## 
## @details
##     - Percent utilization is calculated by taking two snapshots (s1, s2) and
##       using the equation: %util = (s2.activeCounterValue -
##       s1.activeCounterValue) / (s2.samplingCounterValue -
##       s1.samplingCounterValue)
class zes_vf_util_engine_exp2_t(Structure):
    _fields_ = [
        ("stype", zes_structure_type_t),                                ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("vfEngineType", zes_engine_group_t),                           ## [out] The engine group.
        ("activeCounterValue", c_ulonglong),                            ## [out] Represents active counter.
        ("samplingCounterValue", c_ulonglong)                           ## [out] Represents counter value when activeCounterValue was sampled.
                                                                        ## Refer to the formulae above for calculating the utilization percent
    ]

###############################################################################
__use_win_types = "Windows" == platform.uname()[0]

###############################################################################
## @brief Function-pointer for zesInit
if __use_win_types:
    _zesInit_t = WINFUNCTYPE( ze_result_t, zes_init_flags_t )
else:
    _zesInit_t = CFUNCTYPE( ze_result_t, zes_init_flags_t )


###############################################################################
## @brief Table of Global functions pointers
class _zes_global_dditable_t(Structure):
    _fields_ = [
        ("pfnInit", c_void_p)                                           ## _zesInit_t
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
## @brief Function-pointer for zesDeviceEccAvailable
if __use_win_types:
    _zesDeviceEccAvailable_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(ze_bool_t) )
else:
    _zesDeviceEccAvailable_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(ze_bool_t) )

###############################################################################
## @brief Function-pointer for zesDeviceEccConfigurable
if __use_win_types:
    _zesDeviceEccConfigurable_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(ze_bool_t) )
else:
    _zesDeviceEccConfigurable_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(ze_bool_t) )

###############################################################################
## @brief Function-pointer for zesDeviceGetEccState
if __use_win_types:
    _zesDeviceGetEccState_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(zes_device_ecc_properties_t) )
else:
    _zesDeviceGetEccState_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(zes_device_ecc_properties_t) )

###############################################################################
## @brief Function-pointer for zesDeviceSetEccState
if __use_win_types:
    _zesDeviceSetEccState_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(zes_device_ecc_desc_t), POINTER(zes_device_ecc_properties_t) )
else:
    _zesDeviceSetEccState_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(zes_device_ecc_desc_t), POINTER(zes_device_ecc_properties_t) )

###############################################################################
## @brief Function-pointer for zesDeviceGet
if __use_win_types:
    _zesDeviceGet_t = WINFUNCTYPE( ze_result_t, zes_driver_handle_t, POINTER(c_ulong), POINTER(zes_device_handle_t) )
else:
    _zesDeviceGet_t = CFUNCTYPE( ze_result_t, zes_driver_handle_t, POINTER(c_ulong), POINTER(zes_device_handle_t) )

###############################################################################
## @brief Function-pointer for zesDeviceSetOverclockWaiver
if __use_win_types:
    _zesDeviceSetOverclockWaiver_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t )
else:
    _zesDeviceSetOverclockWaiver_t = CFUNCTYPE( ze_result_t, zes_device_handle_t )

###############################################################################
## @brief Function-pointer for zesDeviceGetOverclockDomains
if __use_win_types:
    _zesDeviceGetOverclockDomains_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong) )
else:
    _zesDeviceGetOverclockDomains_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong) )

###############################################################################
## @brief Function-pointer for zesDeviceGetOverclockControls
if __use_win_types:
    _zesDeviceGetOverclockControls_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, zes_overclock_domain_t, POINTER(c_ulong) )
else:
    _zesDeviceGetOverclockControls_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, zes_overclock_domain_t, POINTER(c_ulong) )

###############################################################################
## @brief Function-pointer for zesDeviceResetOverclockSettings
if __use_win_types:
    _zesDeviceResetOverclockSettings_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, ze_bool_t )
else:
    _zesDeviceResetOverclockSettings_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, ze_bool_t )

###############################################################################
## @brief Function-pointer for zesDeviceReadOverclockState
if __use_win_types:
    _zesDeviceReadOverclockState_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(zes_overclock_mode_t), POINTER(ze_bool_t), POINTER(ze_bool_t), POINTER(zes_pending_action_t), POINTER(ze_bool_t) )
else:
    _zesDeviceReadOverclockState_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(zes_overclock_mode_t), POINTER(ze_bool_t), POINTER(ze_bool_t), POINTER(zes_pending_action_t), POINTER(ze_bool_t) )

###############################################################################
## @brief Function-pointer for zesDeviceEnumOverclockDomains
if __use_win_types:
    _zesDeviceEnumOverclockDomains_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_overclock_handle_t) )
else:
    _zesDeviceEnumOverclockDomains_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_overclock_handle_t) )

###############################################################################
## @brief Function-pointer for zesDeviceResetExt
if __use_win_types:
    _zesDeviceResetExt_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(zes_reset_properties_t) )
else:
    _zesDeviceResetExt_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(zes_reset_properties_t) )


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
        ("pfnEnumTemperatureSensors", c_void_p),                        ## _zesDeviceEnumTemperatureSensors_t
        ("pfnEccAvailable", c_void_p),                                  ## _zesDeviceEccAvailable_t
        ("pfnEccConfigurable", c_void_p),                               ## _zesDeviceEccConfigurable_t
        ("pfnGetEccState", c_void_p),                                   ## _zesDeviceGetEccState_t
        ("pfnSetEccState", c_void_p),                                   ## _zesDeviceSetEccState_t
        ("pfnGet", c_void_p),                                           ## _zesDeviceGet_t
        ("pfnSetOverclockWaiver", c_void_p),                            ## _zesDeviceSetOverclockWaiver_t
        ("pfnGetOverclockDomains", c_void_p),                           ## _zesDeviceGetOverclockDomains_t
        ("pfnGetOverclockControls", c_void_p),                          ## _zesDeviceGetOverclockControls_t
        ("pfnResetOverclockSettings", c_void_p),                        ## _zesDeviceResetOverclockSettings_t
        ("pfnReadOverclockState", c_void_p),                            ## _zesDeviceReadOverclockState_t
        ("pfnEnumOverclockDomains", c_void_p),                          ## _zesDeviceEnumOverclockDomains_t
        ("pfnResetExt", c_void_p)                                       ## _zesDeviceResetExt_t
    ]

###############################################################################
## @brief Function-pointer for zesDeviceGetSubDevicePropertiesExp
if __use_win_types:
    _zesDeviceGetSubDevicePropertiesExp_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_subdevice_exp_properties_t) )
else:
    _zesDeviceGetSubDevicePropertiesExp_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_subdevice_exp_properties_t) )

###############################################################################
## @brief Function-pointer for zesDeviceEnumActiveVFExp
if __use_win_types:
    _zesDeviceEnumActiveVFExp_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_vf_handle_t) )
else:
    _zesDeviceEnumActiveVFExp_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_vf_handle_t) )

###############################################################################
## @brief Function-pointer for zesDeviceEnumEnabledVFExp
if __use_win_types:
    _zesDeviceEnumEnabledVFExp_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_vf_handle_t) )
else:
    _zesDeviceEnumEnabledVFExp_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, POINTER(c_ulong), POINTER(zes_vf_handle_t) )


###############################################################################
## @brief Table of DeviceExp functions pointers
class _zes_device_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnGetSubDevicePropertiesExp", c_void_p),                     ## _zesDeviceGetSubDevicePropertiesExp_t
        ("pfnEnumActiveVFExp", c_void_p),                               ## _zesDeviceEnumActiveVFExp_t
        ("pfnEnumEnabledVFExp", c_void_p)                               ## _zesDeviceEnumEnabledVFExp_t
    ]

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
## @brief Function-pointer for zesDriverGet
if __use_win_types:
    _zesDriverGet_t = WINFUNCTYPE( ze_result_t, POINTER(c_ulong), POINTER(zes_driver_handle_t) )
else:
    _zesDriverGet_t = CFUNCTYPE( ze_result_t, POINTER(c_ulong), POINTER(zes_driver_handle_t) )

###############################################################################
## @brief Function-pointer for zesDriverGetExtensionProperties
if __use_win_types:
    _zesDriverGetExtensionProperties_t = WINFUNCTYPE( ze_result_t, zes_driver_handle_t, POINTER(c_ulong), POINTER(zes_driver_extension_properties_t) )
else:
    _zesDriverGetExtensionProperties_t = CFUNCTYPE( ze_result_t, zes_driver_handle_t, POINTER(c_ulong), POINTER(zes_driver_extension_properties_t) )

###############################################################################
## @brief Function-pointer for zesDriverGetExtensionFunctionAddress
if __use_win_types:
    _zesDriverGetExtensionFunctionAddress_t = WINFUNCTYPE( ze_result_t, zes_driver_handle_t, c_char_p, POINTER(c_void_p) )
else:
    _zesDriverGetExtensionFunctionAddress_t = CFUNCTYPE( ze_result_t, zes_driver_handle_t, c_char_p, POINTER(c_void_p) )


###############################################################################
## @brief Table of Driver functions pointers
class _zes_driver_dditable_t(Structure):
    _fields_ = [
        ("pfnEventListen", c_void_p),                                   ## _zesDriverEventListen_t
        ("pfnEventListenEx", c_void_p),                                 ## _zesDriverEventListenEx_t
        ("pfnGet", c_void_p),                                           ## _zesDriverGet_t
        ("pfnGetExtensionProperties", c_void_p),                        ## _zesDriverGetExtensionProperties_t
        ("pfnGetExtensionFunctionAddress", c_void_p)                    ## _zesDriverGetExtensionFunctionAddress_t
    ]

###############################################################################
## @brief Function-pointer for zesDriverGetDeviceByUuidExp
if __use_win_types:
    _zesDriverGetDeviceByUuidExp_t = WINFUNCTYPE( ze_result_t, zes_driver_handle_t, zes_uuid_t, POINTER(zes_device_handle_t), POINTER(ze_bool_t), POINTER(c_ulong) )
else:
    _zesDriverGetDeviceByUuidExp_t = CFUNCTYPE( ze_result_t, zes_driver_handle_t, zes_uuid_t, POINTER(zes_device_handle_t), POINTER(ze_bool_t), POINTER(c_ulong) )


###############################################################################
## @brief Table of DriverExp functions pointers
class _zes_driver_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnGetDeviceByUuidExp", c_void_p)                             ## _zesDriverGetDeviceByUuidExp_t
    ]

###############################################################################
## @brief Function-pointer for zesOverclockGetDomainProperties
if __use_win_types:
    _zesOverclockGetDomainProperties_t = WINFUNCTYPE( ze_result_t, zes_overclock_handle_t, POINTER(zes_overclock_properties_t) )
else:
    _zesOverclockGetDomainProperties_t = CFUNCTYPE( ze_result_t, zes_overclock_handle_t, POINTER(zes_overclock_properties_t) )

###############################################################################
## @brief Function-pointer for zesOverclockGetDomainVFProperties
if __use_win_types:
    _zesOverclockGetDomainVFProperties_t = WINFUNCTYPE( ze_result_t, zes_overclock_handle_t, POINTER(zes_vf_property_t) )
else:
    _zesOverclockGetDomainVFProperties_t = CFUNCTYPE( ze_result_t, zes_overclock_handle_t, POINTER(zes_vf_property_t) )

###############################################################################
## @brief Function-pointer for zesOverclockGetDomainControlProperties
if __use_win_types:
    _zesOverclockGetDomainControlProperties_t = WINFUNCTYPE( ze_result_t, zes_overclock_handle_t, zes_overclock_control_t, POINTER(zes_control_property_t) )
else:
    _zesOverclockGetDomainControlProperties_t = CFUNCTYPE( ze_result_t, zes_overclock_handle_t, zes_overclock_control_t, POINTER(zes_control_property_t) )

###############################################################################
## @brief Function-pointer for zesOverclockGetControlCurrentValue
if __use_win_types:
    _zesOverclockGetControlCurrentValue_t = WINFUNCTYPE( ze_result_t, zes_overclock_handle_t, zes_overclock_control_t, POINTER(c_double) )
else:
    _zesOverclockGetControlCurrentValue_t = CFUNCTYPE( ze_result_t, zes_overclock_handle_t, zes_overclock_control_t, POINTER(c_double) )

###############################################################################
## @brief Function-pointer for zesOverclockGetControlPendingValue
if __use_win_types:
    _zesOverclockGetControlPendingValue_t = WINFUNCTYPE( ze_result_t, zes_overclock_handle_t, zes_overclock_control_t, POINTER(c_double) )
else:
    _zesOverclockGetControlPendingValue_t = CFUNCTYPE( ze_result_t, zes_overclock_handle_t, zes_overclock_control_t, POINTER(c_double) )

###############################################################################
## @brief Function-pointer for zesOverclockSetControlUserValue
if __use_win_types:
    _zesOverclockSetControlUserValue_t = WINFUNCTYPE( ze_result_t, zes_overclock_handle_t, zes_overclock_control_t, c_double, POINTER(zes_pending_action_t) )
else:
    _zesOverclockSetControlUserValue_t = CFUNCTYPE( ze_result_t, zes_overclock_handle_t, zes_overclock_control_t, c_double, POINTER(zes_pending_action_t) )

###############################################################################
## @brief Function-pointer for zesOverclockGetControlState
if __use_win_types:
    _zesOverclockGetControlState_t = WINFUNCTYPE( ze_result_t, zes_overclock_handle_t, zes_overclock_control_t, POINTER(zes_control_state_t), POINTER(zes_pending_action_t) )
else:
    _zesOverclockGetControlState_t = CFUNCTYPE( ze_result_t, zes_overclock_handle_t, zes_overclock_control_t, POINTER(zes_control_state_t), POINTER(zes_pending_action_t) )

###############################################################################
## @brief Function-pointer for zesOverclockGetVFPointValues
if __use_win_types:
    _zesOverclockGetVFPointValues_t = WINFUNCTYPE( ze_result_t, zes_overclock_handle_t, zes_vf_type_t, zes_vf_array_type_t, c_ulong, POINTER(c_ulong) )
else:
    _zesOverclockGetVFPointValues_t = CFUNCTYPE( ze_result_t, zes_overclock_handle_t, zes_vf_type_t, zes_vf_array_type_t, c_ulong, POINTER(c_ulong) )

###############################################################################
## @brief Function-pointer for zesOverclockSetVFPointValues
if __use_win_types:
    _zesOverclockSetVFPointValues_t = WINFUNCTYPE( ze_result_t, zes_overclock_handle_t, zes_vf_type_t, c_ulong, c_ulong )
else:
    _zesOverclockSetVFPointValues_t = CFUNCTYPE( ze_result_t, zes_overclock_handle_t, zes_vf_type_t, c_ulong, c_ulong )


###############################################################################
## @brief Table of Overclock functions pointers
class _zes_overclock_dditable_t(Structure):
    _fields_ = [
        ("pfnGetDomainProperties", c_void_p),                           ## _zesOverclockGetDomainProperties_t
        ("pfnGetDomainVFProperties", c_void_p),                         ## _zesOverclockGetDomainVFProperties_t
        ("pfnGetDomainControlProperties", c_void_p),                    ## _zesOverclockGetDomainControlProperties_t
        ("pfnGetControlCurrentValue", c_void_p),                        ## _zesOverclockGetControlCurrentValue_t
        ("pfnGetControlPendingValue", c_void_p),                        ## _zesOverclockGetControlPendingValue_t
        ("pfnSetControlUserValue", c_void_p),                           ## _zesOverclockSetControlUserValue_t
        ("pfnGetControlState", c_void_p),                               ## _zesOverclockGetControlState_t
        ("pfnGetVFPointValues", c_void_p),                              ## _zesOverclockGetVFPointValues_t
        ("pfnSetVFPointValues", c_void_p)                               ## _zesOverclockSetVFPointValues_t
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
## @brief Function-pointer for zesPowerGetLimitsExt
if __use_win_types:
    _zesPowerGetLimitsExt_t = WINFUNCTYPE( ze_result_t, zes_pwr_handle_t, POINTER(c_ulong), POINTER(zes_power_limit_ext_desc_t) )
else:
    _zesPowerGetLimitsExt_t = CFUNCTYPE( ze_result_t, zes_pwr_handle_t, POINTER(c_ulong), POINTER(zes_power_limit_ext_desc_t) )

###############################################################################
## @brief Function-pointer for zesPowerSetLimitsExt
if __use_win_types:
    _zesPowerSetLimitsExt_t = WINFUNCTYPE( ze_result_t, zes_pwr_handle_t, POINTER(c_ulong), POINTER(zes_power_limit_ext_desc_t) )
else:
    _zesPowerSetLimitsExt_t = CFUNCTYPE( ze_result_t, zes_pwr_handle_t, POINTER(c_ulong), POINTER(zes_power_limit_ext_desc_t) )


###############################################################################
## @brief Table of Power functions pointers
class _zes_power_dditable_t(Structure):
    _fields_ = [
        ("pfnGetProperties", c_void_p),                                 ## _zesPowerGetProperties_t
        ("pfnGetEnergyCounter", c_void_p),                              ## _zesPowerGetEnergyCounter_t
        ("pfnGetLimits", c_void_p),                                     ## _zesPowerGetLimits_t
        ("pfnSetLimits", c_void_p),                                     ## _zesPowerSetLimits_t
        ("pfnGetEnergyThreshold", c_void_p),                            ## _zesPowerGetEnergyThreshold_t
        ("pfnSetEnergyThreshold", c_void_p),                            ## _zesPowerSetEnergyThreshold_t
        ("pfnGetLimitsExt", c_void_p),                                  ## _zesPowerGetLimitsExt_t
        ("pfnSetLimitsExt", c_void_p)                                   ## _zesPowerSetLimitsExt_t
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
## @brief Function-pointer for zesEngineGetActivityExt
if __use_win_types:
    _zesEngineGetActivityExt_t = WINFUNCTYPE( ze_result_t, zes_engine_handle_t, POINTER(c_ulong), POINTER(zes_engine_stats_t) )
else:
    _zesEngineGetActivityExt_t = CFUNCTYPE( ze_result_t, zes_engine_handle_t, POINTER(c_ulong), POINTER(zes_engine_stats_t) )


###############################################################################
## @brief Table of Engine functions pointers
class _zes_engine_dditable_t(Structure):
    _fields_ = [
        ("pfnGetProperties", c_void_p),                                 ## _zesEngineGetProperties_t
        ("pfnGetActivity", c_void_p),                                   ## _zesEngineGetActivity_t
        ("pfnGetActivityExt", c_void_p)                                 ## _zesEngineGetActivityExt_t
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
## @brief Function-pointer for zesFirmwareGetFlashProgress
if __use_win_types:
    _zesFirmwareGetFlashProgress_t = WINFUNCTYPE( ze_result_t, zes_firmware_handle_t, POINTER(c_ulong) )
else:
    _zesFirmwareGetFlashProgress_t = CFUNCTYPE( ze_result_t, zes_firmware_handle_t, POINTER(c_ulong) )

###############################################################################
## @brief Function-pointer for zesFirmwareGetConsoleLogs
if __use_win_types:
    _zesFirmwareGetConsoleLogs_t = WINFUNCTYPE( ze_result_t, zes_firmware_handle_t, POINTER(c_size_t), c_char_p )
else:
    _zesFirmwareGetConsoleLogs_t = CFUNCTYPE( ze_result_t, zes_firmware_handle_t, POINTER(c_size_t), c_char_p )


###############################################################################
## @brief Table of Firmware functions pointers
class _zes_firmware_dditable_t(Structure):
    _fields_ = [
        ("pfnGetProperties", c_void_p),                                 ## _zesFirmwareGetProperties_t
        ("pfnFlash", c_void_p),                                         ## _zesFirmwareFlash_t
        ("pfnGetFlashProgress", c_void_p),                              ## _zesFirmwareGetFlashProgress_t
        ("pfnGetConsoleLogs", c_void_p)                                 ## _zesFirmwareGetConsoleLogs_t
    ]

###############################################################################
## @brief Function-pointer for zesFirmwareGetSecurityVersionExp
if __use_win_types:
    _zesFirmwareGetSecurityVersionExp_t = WINFUNCTYPE( ze_result_t, zes_firmware_handle_t, c_char_p )
else:
    _zesFirmwareGetSecurityVersionExp_t = CFUNCTYPE( ze_result_t, zes_firmware_handle_t, c_char_p )

###############################################################################
## @brief Function-pointer for zesFirmwareSetSecurityVersionExp
if __use_win_types:
    _zesFirmwareSetSecurityVersionExp_t = WINFUNCTYPE( ze_result_t, zes_firmware_handle_t )
else:
    _zesFirmwareSetSecurityVersionExp_t = CFUNCTYPE( ze_result_t, zes_firmware_handle_t )


###############################################################################
## @brief Table of FirmwareExp functions pointers
class _zes_firmware_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnGetSecurityVersionExp", c_void_p),                         ## _zesFirmwareGetSecurityVersionExp_t
        ("pfnSetSecurityVersionExp", c_void_p)                          ## _zesFirmwareSetSecurityVersionExp_t
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
## @brief Function-pointer for zesFabricPortGetFabricErrorCounters
if __use_win_types:
    _zesFabricPortGetFabricErrorCounters_t = WINFUNCTYPE( ze_result_t, zes_fabric_port_handle_t, POINTER(zes_fabric_port_error_counters_t) )
else:
    _zesFabricPortGetFabricErrorCounters_t = CFUNCTYPE( ze_result_t, zes_fabric_port_handle_t, POINTER(zes_fabric_port_error_counters_t) )

###############################################################################
## @brief Function-pointer for zesFabricPortGetMultiPortThroughput
if __use_win_types:
    _zesFabricPortGetMultiPortThroughput_t = WINFUNCTYPE( ze_result_t, zes_device_handle_t, c_ulong, POINTER(zes_fabric_port_handle_t), POINTER(zes_fabric_port_throughput_t*) )
else:
    _zesFabricPortGetMultiPortThroughput_t = CFUNCTYPE( ze_result_t, zes_device_handle_t, c_ulong, POINTER(zes_fabric_port_handle_t), POINTER(zes_fabric_port_throughput_t*) )


###############################################################################
## @brief Table of FabricPort functions pointers
class _zes_fabric_port_dditable_t(Structure):
    _fields_ = [
        ("pfnGetProperties", c_void_p),                                 ## _zesFabricPortGetProperties_t
        ("pfnGetLinkType", c_void_p),                                   ## _zesFabricPortGetLinkType_t
        ("pfnGetConfig", c_void_p),                                     ## _zesFabricPortGetConfig_t
        ("pfnSetConfig", c_void_p),                                     ## _zesFabricPortSetConfig_t
        ("pfnGetState", c_void_p),                                      ## _zesFabricPortGetState_t
        ("pfnGetThroughput", c_void_p),                                 ## _zesFabricPortGetThroughput_t
        ("pfnGetFabricErrorCounters", c_void_p),                        ## _zesFabricPortGetFabricErrorCounters_t
        ("pfnGetMultiPortThroughput", c_void_p)                         ## _zesFabricPortGetMultiPortThroughput_t
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
## @brief Function-pointer for zesRasGetStateExp
if __use_win_types:
    _zesRasGetStateExp_t = WINFUNCTYPE( ze_result_t, zes_ras_handle_t, POINTER(c_ulong), POINTER(zes_ras_state_exp_t) )
else:
    _zesRasGetStateExp_t = CFUNCTYPE( ze_result_t, zes_ras_handle_t, POINTER(c_ulong), POINTER(zes_ras_state_exp_t) )

###############################################################################
## @brief Function-pointer for zesRasClearStateExp
if __use_win_types:
    _zesRasClearStateExp_t = WINFUNCTYPE( ze_result_t, zes_ras_handle_t, zes_ras_error_category_exp_t )
else:
    _zesRasClearStateExp_t = CFUNCTYPE( ze_result_t, zes_ras_handle_t, zes_ras_error_category_exp_t )


###############################################################################
## @brief Table of RasExp functions pointers
class _zes_ras_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnGetStateExp", c_void_p),                                   ## _zesRasGetStateExp_t
        ("pfnClearStateExp", c_void_p)                                  ## _zesRasClearStateExp_t
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
## @brief Function-pointer for zesVFManagementGetVFPropertiesExp
if __use_win_types:
    _zesVFManagementGetVFPropertiesExp_t = WINFUNCTYPE( ze_result_t, zes_vf_handle_t, POINTER(zes_vf_exp_properties_t) )
else:
    _zesVFManagementGetVFPropertiesExp_t = CFUNCTYPE( ze_result_t, zes_vf_handle_t, POINTER(zes_vf_exp_properties_t) )

###############################################################################
## @brief Function-pointer for zesVFManagementGetVFMemoryUtilizationExp
if __use_win_types:
    _zesVFManagementGetVFMemoryUtilizationExp_t = WINFUNCTYPE( ze_result_t, zes_vf_handle_t, POINTER(c_ulong), POINTER(zes_vf_util_mem_exp_t) )
else:
    _zesVFManagementGetVFMemoryUtilizationExp_t = CFUNCTYPE( ze_result_t, zes_vf_handle_t, POINTER(c_ulong), POINTER(zes_vf_util_mem_exp_t) )

###############################################################################
## @brief Function-pointer for zesVFManagementGetVFEngineUtilizationExp
if __use_win_types:
    _zesVFManagementGetVFEngineUtilizationExp_t = WINFUNCTYPE( ze_result_t, zes_vf_handle_t, POINTER(c_ulong), POINTER(zes_vf_util_engine_exp_t) )
else:
    _zesVFManagementGetVFEngineUtilizationExp_t = CFUNCTYPE( ze_result_t, zes_vf_handle_t, POINTER(c_ulong), POINTER(zes_vf_util_engine_exp_t) )

###############################################################################
## @brief Function-pointer for zesVFManagementSetVFTelemetryModeExp
if __use_win_types:
    _zesVFManagementSetVFTelemetryModeExp_t = WINFUNCTYPE( ze_result_t, zes_vf_handle_t, zes_vf_info_util_exp_flags_t, ze_bool_t )
else:
    _zesVFManagementSetVFTelemetryModeExp_t = CFUNCTYPE( ze_result_t, zes_vf_handle_t, zes_vf_info_util_exp_flags_t, ze_bool_t )

###############################################################################
## @brief Function-pointer for zesVFManagementSetVFTelemetrySamplingIntervalExp
if __use_win_types:
    _zesVFManagementSetVFTelemetrySamplingIntervalExp_t = WINFUNCTYPE( ze_result_t, zes_vf_handle_t, zes_vf_info_util_exp_flags_t, c_ulonglong )
else:
    _zesVFManagementSetVFTelemetrySamplingIntervalExp_t = CFUNCTYPE( ze_result_t, zes_vf_handle_t, zes_vf_info_util_exp_flags_t, c_ulonglong )

###############################################################################
## @brief Function-pointer for zesVFManagementGetVFCapabilitiesExp
if __use_win_types:
    _zesVFManagementGetVFCapabilitiesExp_t = WINFUNCTYPE( ze_result_t, zes_vf_handle_t, POINTER(zes_vf_exp_capabilities_t) )
else:
    _zesVFManagementGetVFCapabilitiesExp_t = CFUNCTYPE( ze_result_t, zes_vf_handle_t, POINTER(zes_vf_exp_capabilities_t) )

###############################################################################
## @brief Function-pointer for zesVFManagementGetVFMemoryUtilizationExp2
if __use_win_types:
    _zesVFManagementGetVFMemoryUtilizationExp2_t = WINFUNCTYPE( ze_result_t, zes_vf_handle_t, POINTER(c_ulong), POINTER(zes_vf_util_mem_exp2_t) )
else:
    _zesVFManagementGetVFMemoryUtilizationExp2_t = CFUNCTYPE( ze_result_t, zes_vf_handle_t, POINTER(c_ulong), POINTER(zes_vf_util_mem_exp2_t) )

###############################################################################
## @brief Function-pointer for zesVFManagementGetVFEngineUtilizationExp2
if __use_win_types:
    _zesVFManagementGetVFEngineUtilizationExp2_t = WINFUNCTYPE( ze_result_t, zes_vf_handle_t, POINTER(c_ulong), POINTER(zes_vf_util_engine_exp2_t) )
else:
    _zesVFManagementGetVFEngineUtilizationExp2_t = CFUNCTYPE( ze_result_t, zes_vf_handle_t, POINTER(c_ulong), POINTER(zes_vf_util_engine_exp2_t) )


###############################################################################
## @brief Table of VFManagementExp functions pointers
class _zes_vf_management_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnGetVFPropertiesExp", c_void_p),                            ## _zesVFManagementGetVFPropertiesExp_t
        ("pfnGetVFMemoryUtilizationExp", c_void_p),                     ## _zesVFManagementGetVFMemoryUtilizationExp_t
        ("pfnGetVFEngineUtilizationExp", c_void_p),                     ## _zesVFManagementGetVFEngineUtilizationExp_t
        ("pfnSetVFTelemetryModeExp", c_void_p),                         ## _zesVFManagementSetVFTelemetryModeExp_t
        ("pfnSetVFTelemetrySamplingIntervalExp", c_void_p),             ## _zesVFManagementSetVFTelemetrySamplingIntervalExp_t
        ("pfnGetVFCapabilitiesExp", c_void_p),                          ## _zesVFManagementGetVFCapabilitiesExp_t
        ("pfnGetVFMemoryUtilizationExp2", c_void_p),                    ## _zesVFManagementGetVFMemoryUtilizationExp2_t
        ("pfnGetVFEngineUtilizationExp2", c_void_p)                     ## _zesVFManagementGetVFEngineUtilizationExp2_t
    ]

###############################################################################
class _zes_dditable_t(Structure):
    _fields_ = [
        ("Global", _zes_global_dditable_t),
        ("Device", _zes_device_dditable_t),
        ("DeviceExp", _zes_device_exp_dditable_t),
        ("Driver", _zes_driver_dditable_t),
        ("DriverExp", _zes_driver_exp_dditable_t),
        ("Overclock", _zes_overclock_dditable_t),
        ("Scheduler", _zes_scheduler_dditable_t),
        ("PerformanceFactor", _zes_performance_factor_dditable_t),
        ("Power", _zes_power_dditable_t),
        ("Frequency", _zes_frequency_dditable_t),
        ("Engine", _zes_engine_dditable_t),
        ("Standby", _zes_standby_dditable_t),
        ("Firmware", _zes_firmware_dditable_t),
        ("FirmwareExp", _zes_firmware_exp_dditable_t),
        ("Memory", _zes_memory_dditable_t),
        ("FabricPort", _zes_fabric_port_dditable_t),
        ("Temperature", _zes_temperature_dditable_t),
        ("Psu", _zes_psu_dditable_t),
        ("Fan", _zes_fan_dditable_t),
        ("Led", _zes_led_dditable_t),
        ("Ras", _zes_ras_dditable_t),
        ("RasExp", _zes_ras_exp_dditable_t),
        ("Diagnostics", _zes_diagnostics_dditable_t),
        ("VFManagementExp", _zes_vf_management_exp_dditable_t)
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
        _Global = _zes_global_dditable_t()
        r = ze_result_v(self.__dll.zesGetGlobalProcAddrTable(version, byref(_Global)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Global = _Global

        # attach function interface to function address
        self.zesInit = _zesInit_t(self.__dditable.Global.pfnInit)

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
        self.zesDeviceEccAvailable = _zesDeviceEccAvailable_t(self.__dditable.Device.pfnEccAvailable)
        self.zesDeviceEccConfigurable = _zesDeviceEccConfigurable_t(self.__dditable.Device.pfnEccConfigurable)
        self.zesDeviceGetEccState = _zesDeviceGetEccState_t(self.__dditable.Device.pfnGetEccState)
        self.zesDeviceSetEccState = _zesDeviceSetEccState_t(self.__dditable.Device.pfnSetEccState)
        self.zesDeviceGet = _zesDeviceGet_t(self.__dditable.Device.pfnGet)
        self.zesDeviceSetOverclockWaiver = _zesDeviceSetOverclockWaiver_t(self.__dditable.Device.pfnSetOverclockWaiver)
        self.zesDeviceGetOverclockDomains = _zesDeviceGetOverclockDomains_t(self.__dditable.Device.pfnGetOverclockDomains)
        self.zesDeviceGetOverclockControls = _zesDeviceGetOverclockControls_t(self.__dditable.Device.pfnGetOverclockControls)
        self.zesDeviceResetOverclockSettings = _zesDeviceResetOverclockSettings_t(self.__dditable.Device.pfnResetOverclockSettings)
        self.zesDeviceReadOverclockState = _zesDeviceReadOverclockState_t(self.__dditable.Device.pfnReadOverclockState)
        self.zesDeviceEnumOverclockDomains = _zesDeviceEnumOverclockDomains_t(self.__dditable.Device.pfnEnumOverclockDomains)
        self.zesDeviceResetExt = _zesDeviceResetExt_t(self.__dditable.Device.pfnResetExt)

        # call driver to get function pointers
        _DeviceExp = _zes_device_exp_dditable_t()
        r = ze_result_v(self.__dll.zesGetDeviceExpProcAddrTable(version, byref(_DeviceExp)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.DeviceExp = _DeviceExp

        # attach function interface to function address
        self.zesDeviceGetSubDevicePropertiesExp = _zesDeviceGetSubDevicePropertiesExp_t(self.__dditable.DeviceExp.pfnGetSubDevicePropertiesExp)
        self.zesDeviceEnumActiveVFExp = _zesDeviceEnumActiveVFExp_t(self.__dditable.DeviceExp.pfnEnumActiveVFExp)
        self.zesDeviceEnumEnabledVFExp = _zesDeviceEnumEnabledVFExp_t(self.__dditable.DeviceExp.pfnEnumEnabledVFExp)

        # call driver to get function pointers
        _Driver = _zes_driver_dditable_t()
        r = ze_result_v(self.__dll.zesGetDriverProcAddrTable(version, byref(_Driver)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Driver = _Driver

        # attach function interface to function address
        self.zesDriverEventListen = _zesDriverEventListen_t(self.__dditable.Driver.pfnEventListen)
        self.zesDriverEventListenEx = _zesDriverEventListenEx_t(self.__dditable.Driver.pfnEventListenEx)
        self.zesDriverGet = _zesDriverGet_t(self.__dditable.Driver.pfnGet)
        self.zesDriverGetExtensionProperties = _zesDriverGetExtensionProperties_t(self.__dditable.Driver.pfnGetExtensionProperties)
        self.zesDriverGetExtensionFunctionAddress = _zesDriverGetExtensionFunctionAddress_t(self.__dditable.Driver.pfnGetExtensionFunctionAddress)

        # call driver to get function pointers
        _DriverExp = _zes_driver_exp_dditable_t()
        r = ze_result_v(self.__dll.zesGetDriverExpProcAddrTable(version, byref(_DriverExp)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.DriverExp = _DriverExp

        # attach function interface to function address
        self.zesDriverGetDeviceByUuidExp = _zesDriverGetDeviceByUuidExp_t(self.__dditable.DriverExp.pfnGetDeviceByUuidExp)

        # call driver to get function pointers
        _Overclock = _zes_overclock_dditable_t()
        r = ze_result_v(self.__dll.zesGetOverclockProcAddrTable(version, byref(_Overclock)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Overclock = _Overclock

        # attach function interface to function address
        self.zesOverclockGetDomainProperties = _zesOverclockGetDomainProperties_t(self.__dditable.Overclock.pfnGetDomainProperties)
        self.zesOverclockGetDomainVFProperties = _zesOverclockGetDomainVFProperties_t(self.__dditable.Overclock.pfnGetDomainVFProperties)
        self.zesOverclockGetDomainControlProperties = _zesOverclockGetDomainControlProperties_t(self.__dditable.Overclock.pfnGetDomainControlProperties)
        self.zesOverclockGetControlCurrentValue = _zesOverclockGetControlCurrentValue_t(self.__dditable.Overclock.pfnGetControlCurrentValue)
        self.zesOverclockGetControlPendingValue = _zesOverclockGetControlPendingValue_t(self.__dditable.Overclock.pfnGetControlPendingValue)
        self.zesOverclockSetControlUserValue = _zesOverclockSetControlUserValue_t(self.__dditable.Overclock.pfnSetControlUserValue)
        self.zesOverclockGetControlState = _zesOverclockGetControlState_t(self.__dditable.Overclock.pfnGetControlState)
        self.zesOverclockGetVFPointValues = _zesOverclockGetVFPointValues_t(self.__dditable.Overclock.pfnGetVFPointValues)
        self.zesOverclockSetVFPointValues = _zesOverclockSetVFPointValues_t(self.__dditable.Overclock.pfnSetVFPointValues)

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
        self.zesPowerGetLimitsExt = _zesPowerGetLimitsExt_t(self.__dditable.Power.pfnGetLimitsExt)
        self.zesPowerSetLimitsExt = _zesPowerSetLimitsExt_t(self.__dditable.Power.pfnSetLimitsExt)

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
        self.zesEngineGetActivityExt = _zesEngineGetActivityExt_t(self.__dditable.Engine.pfnGetActivityExt)

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
        self.zesFirmwareGetFlashProgress = _zesFirmwareGetFlashProgress_t(self.__dditable.Firmware.pfnGetFlashProgress)
        self.zesFirmwareGetConsoleLogs = _zesFirmwareGetConsoleLogs_t(self.__dditable.Firmware.pfnGetConsoleLogs)

        # call driver to get function pointers
        _FirmwareExp = _zes_firmware_exp_dditable_t()
        r = ze_result_v(self.__dll.zesGetFirmwareExpProcAddrTable(version, byref(_FirmwareExp)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.FirmwareExp = _FirmwareExp

        # attach function interface to function address
        self.zesFirmwareGetSecurityVersionExp = _zesFirmwareGetSecurityVersionExp_t(self.__dditable.FirmwareExp.pfnGetSecurityVersionExp)
        self.zesFirmwareSetSecurityVersionExp = _zesFirmwareSetSecurityVersionExp_t(self.__dditable.FirmwareExp.pfnSetSecurityVersionExp)

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
        self.zesFabricPortGetFabricErrorCounters = _zesFabricPortGetFabricErrorCounters_t(self.__dditable.FabricPort.pfnGetFabricErrorCounters)
        self.zesFabricPortGetMultiPortThroughput = _zesFabricPortGetMultiPortThroughput_t(self.__dditable.FabricPort.pfnGetMultiPortThroughput)

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
        _RasExp = _zes_ras_exp_dditable_t()
        r = ze_result_v(self.__dll.zesGetRasExpProcAddrTable(version, byref(_RasExp)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.RasExp = _RasExp

        # attach function interface to function address
        self.zesRasGetStateExp = _zesRasGetStateExp_t(self.__dditable.RasExp.pfnGetStateExp)
        self.zesRasClearStateExp = _zesRasClearStateExp_t(self.__dditable.RasExp.pfnClearStateExp)

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

        # call driver to get function pointers
        _VFManagementExp = _zes_vf_management_exp_dditable_t()
        r = ze_result_v(self.__dll.zesGetVFManagementExpProcAddrTable(version, byref(_VFManagementExp)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.VFManagementExp = _VFManagementExp

        # attach function interface to function address
        self.zesVFManagementGetVFPropertiesExp = _zesVFManagementGetVFPropertiesExp_t(self.__dditable.VFManagementExp.pfnGetVFPropertiesExp)
        self.zesVFManagementGetVFMemoryUtilizationExp = _zesVFManagementGetVFMemoryUtilizationExp_t(self.__dditable.VFManagementExp.pfnGetVFMemoryUtilizationExp)
        self.zesVFManagementGetVFEngineUtilizationExp = _zesVFManagementGetVFEngineUtilizationExp_t(self.__dditable.VFManagementExp.pfnGetVFEngineUtilizationExp)
        self.zesVFManagementSetVFTelemetryModeExp = _zesVFManagementSetVFTelemetryModeExp_t(self.__dditable.VFManagementExp.pfnSetVFTelemetryModeExp)
        self.zesVFManagementSetVFTelemetrySamplingIntervalExp = _zesVFManagementSetVFTelemetrySamplingIntervalExp_t(self.__dditable.VFManagementExp.pfnSetVFTelemetrySamplingIntervalExp)
        self.zesVFManagementGetVFCapabilitiesExp = _zesVFManagementGetVFCapabilitiesExp_t(self.__dditable.VFManagementExp.pfnGetVFCapabilitiesExp)
        self.zesVFManagementGetVFMemoryUtilizationExp2 = _zesVFManagementGetVFMemoryUtilizationExp2_t(self.__dditable.VFManagementExp.pfnGetVFMemoryUtilizationExp2)
        self.zesVFManagementGetVFEngineUtilizationExp2 = _zesVFManagementGetVFEngineUtilizationExp2_t(self.__dditable.VFManagementExp.pfnGetVFEngineUtilizationExp2)

        # success!
