"""
 Copyright (C) 2019-2021 Intel Corporation

 SPDX-License-Identifier: MIT

 @file ze.py
 @version v1.11-r1.11.8

 """
import platform
from ctypes import *
from enum import *

###############################################################################
__version__ = "1.0"

###############################################################################
## @brief Generates generic 'oneAPI' API versions
def ZE_MAKE_VERSION( _major, _minor ):
    return (( _major << 16 )|( _minor & 0x0000ffff))

###############################################################################
## @brief Extracts 'oneAPI' API major version
def ZE_MAJOR_VERSION( _ver ):
    return ( _ver >> 16 )

###############################################################################
## @brief Extracts 'oneAPI' API minor version
def ZE_MINOR_VERSION( _ver ):
    return ( _ver & 0x0000ffff )

###############################################################################
## @brief Calling convention for all API functions
# ZE_APICALL not required for python

###############################################################################
## @brief Microsoft-specific dllexport storage-class attribute
# ZE_APIEXPORT not required for python

###############################################################################
## @brief GCC-specific dllexport storage-class attribute
# ZE_APIEXPORT not required for python

###############################################################################
## @brief Microsoft-specific dllexport storage-class attribute
# ZE_DLLEXPORT not required for python

###############################################################################
## @brief GCC-specific dllexport storage-class attribute
# ZE_DLLEXPORT not required for python

###############################################################################
## @brief compiler-independent type
class ze_bool_t(c_ubyte):
    pass

###############################################################################
## @brief Handle of a driver instance
class ze_driver_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of driver's device object
class ze_device_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of driver's context object
class ze_context_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of driver's command queue object
class ze_command_queue_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of driver's command list object
class ze_command_list_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of driver's fence object
class ze_fence_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of driver's event pool object
class ze_event_pool_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of driver's event object
class ze_event_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of driver's image object
class ze_image_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of driver's module object
class ze_module_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of module's build log object
class ze_module_build_log_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of driver's kernel object
class ze_kernel_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of driver's sampler object
class ze_sampler_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of physical memory object
class ze_physical_mem_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of driver's fabric vertex object
class ze_fabric_vertex_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of driver's fabric edge object
class ze_fabric_edge_handle_t(c_void_p):
    pass

###############################################################################
## @brief Maximum IPC handle size
ZE_MAX_IPC_HANDLE_SIZE = 64

###############################################################################
## @brief IPC handle to a memory allocation
class ze_ipc_mem_handle_t(Structure):
    _fields_ = [
        ("data", c_char * ZE_MAX_IPC_HANDLE_SIZE)                       ## [out] Opaque data representing an IPC handle
    ]

###############################################################################
## @brief IPC handle to a event pool allocation
class ze_ipc_event_pool_handle_t(Structure):
    _fields_ = [
        ("data", c_char * ZE_MAX_IPC_HANDLE_SIZE)                       ## [out] Opaque data representing an IPC handle
    ]

###############################################################################
## @brief Generic macro for enumerator bit masks
def ZE_BIT( _i ):
    return ( 1 << _i )

###############################################################################
## @brief Defines Return/Error codes
class ze_result_v(IntEnum):
    SUCCESS = 0                                                             ## [Core] success
    NOT_READY = 1                                                           ## [Core] synchronization primitive not signaled
    ERROR_DEVICE_LOST = 0x70000001                                          ## [Core] device hung, reset, was removed, or driver update occurred
    ERROR_OUT_OF_HOST_MEMORY = 0x70000002                                   ## [Core] insufficient host memory to satisfy call
    ERROR_OUT_OF_DEVICE_MEMORY = 0x70000003                                 ## [Core] insufficient device memory to satisfy call
    ERROR_MODULE_BUILD_FAILURE = 0x70000004                                 ## [Core] error occurred when building module, see build log for details
    ERROR_MODULE_LINK_FAILURE = 0x70000005                                  ## [Core] error occurred when linking modules, see build log for details
    ERROR_DEVICE_REQUIRES_RESET = 0x70000006                                ## [Core] device requires a reset
    ERROR_DEVICE_IN_LOW_POWER_STATE = 0x70000007                            ## [Core] device currently in low power state
    EXP_ERROR_DEVICE_IS_NOT_VERTEX = 0x7ff00001                             ## [Core, Experimental] device is not represented by a fabric vertex
    EXP_ERROR_VERTEX_IS_NOT_DEVICE = 0x7ff00002                             ## [Core, Experimental] fabric vertex does not represent a device
    EXP_ERROR_REMOTE_DEVICE = 0x7ff00003                                    ## [Core, Experimental] fabric vertex represents a remote device or
                                                                            ## subdevice
    EXP_ERROR_OPERANDS_INCOMPATIBLE = 0x7ff00004                            ## [Core, Experimental] operands of comparison are not compatible
    EXP_RTAS_BUILD_RETRY = 0x7ff00005                                       ## [Core, Experimental] ray tracing acceleration structure build
                                                                            ## operation failed due to insufficient resources, retry with a larger
                                                                            ## acceleration structure buffer allocation
    EXP_RTAS_BUILD_DEFERRED = 0x7ff00006                                    ## [Core, Experimental] ray tracing acceleration structure build
                                                                            ## operation deferred to parallel operation join
    ERROR_INSUFFICIENT_PERMISSIONS = 0x70010000                             ## [Sysman] access denied due to permission level
    ERROR_NOT_AVAILABLE = 0x70010001                                        ## [Sysman] resource already in use and simultaneous access not allowed
                                                                            ## or resource was removed
    ERROR_DEPENDENCY_UNAVAILABLE = 0x70020000                               ## [Common] external required dependency is unavailable or missing
    WARNING_DROPPED_DATA = 0x70020001                                       ## [Tools] data may have been dropped
    ERROR_UNINITIALIZED = 0x78000001                                        ## [Validation] driver is not initialized
    ERROR_UNSUPPORTED_VERSION = 0x78000002                                  ## [Validation] generic error code for unsupported versions
    ERROR_UNSUPPORTED_FEATURE = 0x78000003                                  ## [Validation] generic error code for unsupported features
    ERROR_INVALID_ARGUMENT = 0x78000004                                     ## [Validation] generic error code for invalid arguments
    ERROR_INVALID_NULL_HANDLE = 0x78000005                                  ## [Validation] handle argument is not valid
    ERROR_HANDLE_OBJECT_IN_USE = 0x78000006                                 ## [Validation] object pointed to by handle still in-use by device
    ERROR_INVALID_NULL_POINTER = 0x78000007                                 ## [Validation] pointer argument may not be nullptr
    ERROR_INVALID_SIZE = 0x78000008                                         ## [Validation] size argument is invalid (e.g., must not be zero)
    ERROR_UNSUPPORTED_SIZE = 0x78000009                                     ## [Validation] size argument is not supported by the device (e.g., too
                                                                            ## large)
    ERROR_UNSUPPORTED_ALIGNMENT = 0x7800000a                                ## [Validation] alignment argument is not supported by the device (e.g.,
                                                                            ## too small)
    ERROR_INVALID_SYNCHRONIZATION_OBJECT = 0x7800000b                       ## [Validation] synchronization object in invalid state
    ERROR_INVALID_ENUMERATION = 0x7800000c                                  ## [Validation] enumerator argument is not valid
    ERROR_UNSUPPORTED_ENUMERATION = 0x7800000d                              ## [Validation] enumerator argument is not supported by the device
    ERROR_UNSUPPORTED_IMAGE_FORMAT = 0x7800000e                             ## [Validation] image format is not supported by the device
    ERROR_INVALID_NATIVE_BINARY = 0x7800000f                                ## [Validation] native binary is not supported by the device
    ERROR_INVALID_GLOBAL_NAME = 0x78000010                                  ## [Validation] global variable is not found in the module
    ERROR_INVALID_KERNEL_NAME = 0x78000011                                  ## [Validation] kernel name is not found in the module
    ERROR_INVALID_FUNCTION_NAME = 0x78000012                                ## [Validation] function name is not found in the module
    ERROR_INVALID_GROUP_SIZE_DIMENSION = 0x78000013                         ## [Validation] group size dimension is not valid for the kernel or
                                                                            ## device
    ERROR_INVALID_GLOBAL_WIDTH_DIMENSION = 0x78000014                       ## [Validation] global width dimension is not valid for the kernel or
                                                                            ## device
    ERROR_INVALID_KERNEL_ARGUMENT_INDEX = 0x78000015                        ## [Validation] kernel argument index is not valid for kernel
    ERROR_INVALID_KERNEL_ARGUMENT_SIZE = 0x78000016                         ## [Validation] kernel argument size does not match kernel
    ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE = 0x78000017                       ## [Validation] value of kernel attribute is not valid for the kernel or
                                                                            ## device
    ERROR_INVALID_MODULE_UNLINKED = 0x78000018                              ## [Validation] module with imports needs to be linked before kernels can
                                                                            ## be created from it.
    ERROR_INVALID_COMMAND_LIST_TYPE = 0x78000019                            ## [Validation] command list type does not match command queue type
    ERROR_OVERLAPPING_REGIONS = 0x7800001a                                  ## [Validation] copy operations do not support overlapping regions of
                                                                            ## memory
    WARNING_ACTION_REQUIRED = 0x7800001b                                    ## [Sysman] an action is required to complete the desired operation
    ERROR_INVALID_KERNEL_HANDLE = 0x7800001c                                ## [Core, Validation] kernel handle is invalid for the operation
    ERROR_UNKNOWN = 0x7ffffffe                                              ## [Core] unknown or internal error

class ze_result_t(c_int):
    def __str__(self):
        return str(ze_result_v(self.value))


###############################################################################
## @brief Defines structure types
class ze_structure_type_v(IntEnum):
    DRIVER_PROPERTIES = 0x1                                                 ## ::ze_driver_properties_t
    DRIVER_IPC_PROPERTIES = 0x2                                             ## ::ze_driver_ipc_properties_t
    DEVICE_PROPERTIES = 0x3                                                 ## ::ze_device_properties_t
    DEVICE_COMPUTE_PROPERTIES = 0x4                                         ## ::ze_device_compute_properties_t
    DEVICE_MODULE_PROPERTIES = 0x5                                          ## ::ze_device_module_properties_t
    COMMAND_QUEUE_GROUP_PROPERTIES = 0x6                                    ## ::ze_command_queue_group_properties_t
    DEVICE_MEMORY_PROPERTIES = 0x7                                          ## ::ze_device_memory_properties_t
    DEVICE_MEMORY_ACCESS_PROPERTIES = 0x8                                   ## ::ze_device_memory_access_properties_t
    DEVICE_CACHE_PROPERTIES = 0x9                                           ## ::ze_device_cache_properties_t
    DEVICE_IMAGE_PROPERTIES = 0xa                                           ## ::ze_device_image_properties_t
    DEVICE_P2P_PROPERTIES = 0xb                                             ## ::ze_device_p2p_properties_t
    DEVICE_EXTERNAL_MEMORY_PROPERTIES = 0xc                                 ## ::ze_device_external_memory_properties_t
    CONTEXT_DESC = 0xd                                                      ## ::ze_context_desc_t
    COMMAND_QUEUE_DESC = 0xe                                                ## ::ze_command_queue_desc_t
    COMMAND_LIST_DESC = 0xf                                                 ## ::ze_command_list_desc_t
    EVENT_POOL_DESC = 0x10                                                  ## ::ze_event_pool_desc_t
    EVENT_DESC = 0x11                                                       ## ::ze_event_desc_t
    FENCE_DESC = 0x12                                                       ## ::ze_fence_desc_t
    IMAGE_DESC = 0x13                                                       ## ::ze_image_desc_t
    IMAGE_PROPERTIES = 0x14                                                 ## ::ze_image_properties_t
    DEVICE_MEM_ALLOC_DESC = 0x15                                            ## ::ze_device_mem_alloc_desc_t
    HOST_MEM_ALLOC_DESC = 0x16                                              ## ::ze_host_mem_alloc_desc_t
    MEMORY_ALLOCATION_PROPERTIES = 0x17                                     ## ::ze_memory_allocation_properties_t
    EXTERNAL_MEMORY_EXPORT_DESC = 0x18                                      ## ::ze_external_memory_export_desc_t
    EXTERNAL_MEMORY_IMPORT_FD = 0x19                                        ## ::ze_external_memory_import_fd_t
    EXTERNAL_MEMORY_EXPORT_FD = 0x1a                                        ## ::ze_external_memory_export_fd_t
    MODULE_DESC = 0x1b                                                      ## ::ze_module_desc_t
    MODULE_PROPERTIES = 0x1c                                                ## ::ze_module_properties_t
    KERNEL_DESC = 0x1d                                                      ## ::ze_kernel_desc_t
    KERNEL_PROPERTIES = 0x1e                                                ## ::ze_kernel_properties_t
    SAMPLER_DESC = 0x1f                                                     ## ::ze_sampler_desc_t
    PHYSICAL_MEM_DESC = 0x20                                                ## ::ze_physical_mem_desc_t
    KERNEL_PREFERRED_GROUP_SIZE_PROPERTIES = 0x21                           ## ::ze_kernel_preferred_group_size_properties_t
    EXTERNAL_MEMORY_IMPORT_WIN32 = 0x22                                     ## ::ze_external_memory_import_win32_handle_t
    EXTERNAL_MEMORY_EXPORT_WIN32 = 0x23                                     ## ::ze_external_memory_export_win32_handle_t
    DEVICE_RAYTRACING_EXT_PROPERTIES = 0x00010001                           ## ::ze_device_raytracing_ext_properties_t
    RAYTRACING_MEM_ALLOC_EXT_DESC = 0x10002                                 ## ::ze_raytracing_mem_alloc_ext_desc_t
    FLOAT_ATOMIC_EXT_PROPERTIES = 0x10003                                   ## ::ze_float_atomic_ext_properties_t
    CACHE_RESERVATION_EXT_DESC = 0x10004                                    ## ::ze_cache_reservation_ext_desc_t
    EU_COUNT_EXT = 0x10005                                                  ## ::ze_eu_count_ext_t
    SRGB_EXT_DESC = 0x10006                                                 ## ::ze_srgb_ext_desc_t
    LINKAGE_INSPECTION_EXT_DESC = 0x10007                                   ## ::ze_linkage_inspection_ext_desc_t
    PCI_EXT_PROPERTIES = 0x10008                                            ## ::ze_pci_ext_properties_t
    DRIVER_MEMORY_FREE_EXT_PROPERTIES = 0x10009                             ## ::ze_driver_memory_free_ext_properties_t
    MEMORY_FREE_EXT_DESC = 0x1000a                                          ## ::ze_memory_free_ext_desc_t
    MEMORY_COMPRESSION_HINTS_EXT_DESC = 0x1000b                             ## ::ze_memory_compression_hints_ext_desc_t
    IMAGE_ALLOCATION_EXT_PROPERTIES = 0x1000c                               ## ::ze_image_allocation_ext_properties_t
    DEVICE_LUID_EXT_PROPERTIES = 0x1000d                                    ## ::ze_device_luid_ext_properties_t
    DEVICE_MEMORY_EXT_PROPERTIES = 0x1000e                                  ## ::ze_device_memory_ext_properties_t
    DEVICE_IP_VERSION_EXT = 0x1000f                                         ## ::ze_device_ip_version_ext_t
    IMAGE_VIEW_PLANAR_EXT_DESC = 0x10010                                    ## ::ze_image_view_planar_ext_desc_t
    EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_PROPERTIES = 0x10011                  ## ::ze_event_query_kernel_timestamps_ext_properties_t
    EVENT_QUERY_KERNEL_TIMESTAMPS_RESULTS_EXT_PROPERTIES = 0x10012          ## ::ze_event_query_kernel_timestamps_results_ext_properties_t
    KERNEL_MAX_GROUP_SIZE_EXT_PROPERTIES = 0x10013                          ## ::ze_kernel_max_group_size_ext_properties_t
    RELAXED_ALLOCATION_LIMITS_EXP_DESC = 0x00020001                         ## ::ze_relaxed_allocation_limits_exp_desc_t
    MODULE_PROGRAM_EXP_DESC = 0x00020002                                    ## ::ze_module_program_exp_desc_t
    SCHEDULING_HINT_EXP_PROPERTIES = 0x00020003                             ## ::ze_scheduling_hint_exp_properties_t
    SCHEDULING_HINT_EXP_DESC = 0x00020004                                   ## ::ze_scheduling_hint_exp_desc_t
    IMAGE_VIEW_PLANAR_EXP_DESC = 0x00020005                                 ## ::ze_image_view_planar_exp_desc_t
    DEVICE_PROPERTIES_1_2 = 0x00020006                                      ## ::ze_device_properties_t
    IMAGE_MEMORY_EXP_PROPERTIES = 0x00020007                                ## ::ze_image_memory_properties_exp_t
    POWER_SAVING_HINT_EXP_DESC = 0x00020008                                 ## ::ze_context_power_saving_hint_exp_desc_t
    COPY_BANDWIDTH_EXP_PROPERTIES = 0x00020009                              ## ::ze_copy_bandwidth_exp_properties_t
    DEVICE_P2P_BANDWIDTH_EXP_PROPERTIES = 0x0002000A                        ## ::ze_device_p2p_bandwidth_exp_properties_t
    FABRIC_VERTEX_EXP_PROPERTIES = 0x0002000B                               ## ::ze_fabric_vertex_exp_properties_t
    FABRIC_EDGE_EXP_PROPERTIES = 0x0002000C                                 ## ::ze_fabric_edge_exp_properties_t
    MEMORY_SUB_ALLOCATIONS_EXP_PROPERTIES = 0x0002000D                      ## ::ze_memory_sub_allocations_exp_properties_t
    RTAS_BUILDER_EXP_DESC = 0x0002000E                                      ## ::ze_rtas_builder_exp_desc_t
    RTAS_BUILDER_BUILD_OP_EXP_DESC = 0x0002000F                             ## ::ze_rtas_builder_build_op_exp_desc_t
    RTAS_BUILDER_EXP_PROPERTIES = 0x00020010                                ## ::ze_rtas_builder_exp_properties_t
    RTAS_PARALLEL_OPERATION_EXP_PROPERTIES = 0x00020011                     ## ::ze_rtas_parallel_operation_exp_properties_t
    RTAS_DEVICE_EXP_PROPERTIES = 0x00020012                                 ## ::ze_rtas_device_exp_properties_t
    RTAS_GEOMETRY_AABBS_EXP_CB_PARAMS = 0x00020013                          ## ::ze_rtas_geometry_aabbs_exp_cb_params_t
    COUNTER_BASED_EVENT_POOL_EXP_DESC = 0x00020014                          ## ::ze_event_pool_counter_based_exp_desc_t
    MUTABLE_COMMAND_LIST_EXP_PROPERTIES = 0x00020015                        ## ::ze_mutable_command_list_exp_properties_t
    MUTABLE_COMMAND_LIST_EXP_DESC = 0x00020016                              ## ::ze_mutable_command_list_exp_desc_t
    MUTABLE_COMMAND_ID_EXP_DESC = 0x00020017                                ## ::ze_mutable_command_id_exp_desc_t
    MUTABLE_COMMANDS_EXP_DESC = 0x00020018                                  ## ::ze_mutable_commands_exp_desc_t
    MUTABLE_KERNEL_ARGUMENT_EXP_DESC = 0x00020019                           ## ::ze_mutable_kernel_argument_exp_desc_t
    MUTABLE_GROUP_COUNT_EXP_DESC = 0x0002001A                               ## ::ze_mutable_group_count_exp_desc_t
    MUTABLE_GROUP_SIZE_EXP_DESC = 0x0002001B                                ## ::ze_mutable_group_size_exp_desc_t
    MUTABLE_GLOBAL_OFFSET_EXP_DESC = 0x0002001C                             ## ::ze_mutable_global_offset_exp_desc_t
    PITCHED_ALLOC_DEVICE_EXP_PROPERTIES = 0x0002001D                        ## ::ze_device_pitched_alloc_exp_properties_t
    BINDLESS_IMAGE_EXP_DESC = 0x0002001E                                    ## ::ze_image_bindless_exp_desc_t
    PITCHED_IMAGE_EXP_DESC = 0x0002001F                                     ## ::ze_image_pitched_exp_desc_t
    MUTABLE_GRAPH_ARGUMENT_EXP_DESC = 0x00020020                            ## ::ze_mutable_graph_argument_exp_desc_t
    INIT_DRIVER_TYPE_DESC = 0x00020021                                      ## ::ze_init_driver_type_desc_t

class ze_structure_type_t(c_int):
    def __str__(self):
        return str(ze_structure_type_v(self.value))


###############################################################################
## @brief External memory type flags
class ze_external_memory_type_flags_v(IntEnum):
    OPAQUE_FD = ZE_BIT(0)                                                   ## an opaque POSIX file descriptor handle
    DMA_BUF = ZE_BIT(1)                                                     ## a file descriptor handle for a Linux dma_buf
    OPAQUE_WIN32 = ZE_BIT(2)                                                ## an NT handle
    OPAQUE_WIN32_KMT = ZE_BIT(3)                                            ## a global share (KMT) handle
    D3D11_TEXTURE = ZE_BIT(4)                                               ## an NT handle referring to a Direct3D 10 or 11 texture resource
    D3D11_TEXTURE_KMT = ZE_BIT(5)                                           ## a global share (KMT) handle referring to a Direct3D 10 or 11 texture
                                                                            ## resource
    D3D12_HEAP = ZE_BIT(6)                                                  ## an NT handle referring to a Direct3D 12 heap resource
    D3D12_RESOURCE = ZE_BIT(7)                                              ## an NT handle referring to a Direct3D 12 committed resource

class ze_external_memory_type_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Bandwidth unit
class ze_bandwidth_unit_v(IntEnum):
    UNKNOWN = 0                                                             ## The unit used for bandwidth is unknown
    BYTES_PER_NANOSEC = 1                                                   ## Bandwidth is provided in bytes/nanosec
    BYTES_PER_CLOCK = 2                                                     ## Bandwidth is provided in bytes/clock

class ze_bandwidth_unit_t(c_int):
    def __str__(self):
        return str(ze_bandwidth_unit_v(self.value))


###############################################################################
## @brief Latency unit
class ze_latency_unit_v(IntEnum):
    UNKNOWN = 0                                                             ## The unit used for latency is unknown
    NANOSEC = 1                                                             ## Latency is provided in nanosecs
    CLOCK = 2                                                               ## Latency is provided in clocks
    HOP = 3                                                                 ## Latency is provided in hops (normalized so that the lowest latency
                                                                            ## link has a latency of 1 hop)

class ze_latency_unit_t(c_int):
    def __str__(self):
        return str(ze_latency_unit_v(self.value))


###############################################################################
## @brief Maximum universal unique id (UUID) size in bytes
ZE_MAX_UUID_SIZE = 16

###############################################################################
## @brief Universal unique id (UUID)
class ze_uuid_t(Structure):
    _fields_ = [
        ("id", c_ubyte * ZE_MAX_UUID_SIZE)                              ## [out] opaque data representing a UUID
    ]

###############################################################################
## @brief Base for all callback function parameter types
class ze_base_cb_params_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p)                                             ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
    ]

###############################################################################
## @brief Base for all properties types
class ze_base_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p)                                             ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
    ]

###############################################################################
## @brief Base for all descriptor types
class ze_base_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p)                                             ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
    ]

###############################################################################
## @brief Forces driver to only report devices (and sub-devices) as specified by
##        values

###############################################################################
## @brief Forces driver to report devices from lowest to highest PCI bus ID

###############################################################################
## @brief Forces all shared allocations into device memory

###############################################################################
## @brief Defines the device hierarchy model exposed by Level Zero driver
##        implementation

###############################################################################
## @brief Supported initialization flags
class ze_init_flags_v(IntEnum):
    GPU_ONLY = ZE_BIT(0)                                                    ## only initialize GPU drivers
    VPU_ONLY = ZE_BIT(1)                                                    ## only initialize VPU drivers

class ze_init_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Supported driver initialization type flags
## 
## @details
##     - Bit Field which details the driver types to be initialized and
##       returned to the user.
##     - Value Definition:
##     - 0, do not init or retrieve any drivers.
##     - ZE_INIT_DRIVER_TYPE_FLAG_GPU,	GPU Drivers are Init and driver handles
##       retrieved.
##     - ZE_INIT_DRIVER_TYPE_FLAG_NPU,	NPU Drivers are Init and driver handles
##       retrieved.
##     - ZE_INIT_DRIVER_TYPE_FLAG_GPU | ZE_INIT_DRIVER_TYPE_FLAG_NPU, NPU & GPU
##       Drivers are Init and driver handles retrieved.
##     - UINT32_MAX	All Drivers of any type are Init and driver handles
##       retrieved.
class ze_init_driver_type_flags_v(IntEnum):
    GPU = ZE_BIT(0)                                                         ## initialize and retrieve GPU drivers
    NPU = ZE_BIT(1)                                                         ## initialize and retrieve NPU drivers

class ze_init_driver_type_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Init Driver Type descriptor
class ze_init_driver_type_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_init_driver_type_flags_t)                          ## [in] driver type init flags.
                                                                        ## must be a valid combination of ::ze_init_driver_type_flag_t or UINT32_MAX;
                                                                        ## driver types are init and retrieved based on these init flags in zeInitDrivers().
    ]

###############################################################################
## @brief Supported API versions
## 
## @details
##     - API versions contain major and minor attributes, use
##       ::ZE_MAJOR_VERSION and ::ZE_MINOR_VERSION
class ze_api_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    _1_1 = ZE_MAKE_VERSION( 1, 1 )                                          ## version 1.1
    _1_2 = ZE_MAKE_VERSION( 1, 2 )                                          ## version 1.2
    _1_3 = ZE_MAKE_VERSION( 1, 3 )                                          ## version 1.3
    _1_4 = ZE_MAKE_VERSION( 1, 4 )                                          ## version 1.4
    _1_5 = ZE_MAKE_VERSION( 1, 5 )                                          ## version 1.5
    _1_6 = ZE_MAKE_VERSION( 1, 6 )                                          ## version 1.6
    _1_7 = ZE_MAKE_VERSION( 1, 7 )                                          ## version 1.7
    _1_8 = ZE_MAKE_VERSION( 1, 8 )                                          ## version 1.8
    _1_9 = ZE_MAKE_VERSION( 1, 9 )                                          ## version 1.9
    _1_10 = ZE_MAKE_VERSION( 1, 10 )                                        ## version 1.10
    _1_11 = ZE_MAKE_VERSION( 1, 11 )                                        ## version 1.11
    CURRENT = ZE_MAKE_VERSION( 1, 11 )                                      ## latest known version

class ze_api_version_t(c_int):
    def __str__(self):
        return str(ze_api_version_v(self.value))


###############################################################################
## @brief Current API version as a macro
ZE_API_VERSION_CURRENT_M = ZE_MAKE_VERSION( 1, 11 )

###############################################################################
## @brief Maximum driver universal unique id (UUID) size in bytes
ZE_MAX_DRIVER_UUID_SIZE = 16

###############################################################################
## @brief Driver universal unique id (UUID)
class ze_driver_uuid_t(Structure):
    _fields_ = [
        ("id", c_ubyte * ZE_MAX_DRIVER_UUID_SIZE)                       ## [out] opaque data representing a driver UUID
    ]

###############################################################################
## @brief Driver properties queried using ::zeDriverGetProperties
class ze_driver_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("uuid", ze_driver_uuid_t),                                     ## [out] universal unique identifier.
        ("driverVersion", c_ulong)                                      ## [out] driver version
                                                                        ## The driver version is a non-zero, monotonically increasing value where
                                                                        ## higher values always indicate a more recent version.
    ]

###############################################################################
## @brief Supported IPC property flags
class ze_ipc_property_flags_v(IntEnum):
    MEMORY = ZE_BIT(0)                                                      ## Supports passing memory allocations between processes. See
                                                                            ## ::zeMemGetIpcHandle.
    EVENT_POOL = ZE_BIT(1)                                                  ## Supports passing event pools between processes. See
                                                                            ## ::zeEventPoolGetIpcHandle.

class ze_ipc_property_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief IPC properties queried using ::zeDriverGetIpcProperties
class ze_driver_ipc_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_ipc_property_flags_t)                              ## [out] 0 (none) or a valid combination of ::ze_ipc_property_flag_t
    ]

###############################################################################
## @brief Maximum extension name string size
ZE_MAX_EXTENSION_NAME = 256

###############################################################################
## @brief Extension properties queried using ::zeDriverGetExtensionProperties
class ze_driver_extension_properties_t(Structure):
    _fields_ = [
        ("name", c_char * ZE_MAX_EXTENSION_NAME),                       ## [out] extension name
        ("version", c_ulong)                                            ## [out] extension version using ::ZE_MAKE_VERSION
    ]

###############################################################################
## @brief Supported device types
class ze_device_type_v(IntEnum):
    GPU = 1                                                                 ## Graphics Processing Unit
    CPU = 2                                                                 ## Central Processing Unit
    FPGA = 3                                                                ## Field Programmable Gate Array
    MCA = 4                                                                 ## Memory Copy Accelerator
    VPU = 5                                                                 ## Vision Processing Unit

class ze_device_type_t(c_int):
    def __str__(self):
        return str(ze_device_type_v(self.value))


###############################################################################
## @brief Maximum device universal unique id (UUID) size in bytes
ZE_MAX_DEVICE_UUID_SIZE = 16

###############################################################################
## @brief Device universal unique id (UUID)
class ze_device_uuid_t(Structure):
    _fields_ = [
        ("id", c_ubyte * ZE_MAX_DEVICE_UUID_SIZE)                       ## [out] opaque data representing a device UUID
    ]

###############################################################################
## @brief Maximum device name string size
ZE_MAX_DEVICE_NAME = 256

###############################################################################
## @brief Supported device property flags
class ze_device_property_flags_v(IntEnum):
    INTEGRATED = ZE_BIT(0)                                                  ## Device is integrated with the Host.
    SUBDEVICE = ZE_BIT(1)                                                   ## Device handle used for query represents a sub-device.
    ECC = ZE_BIT(2)                                                         ## Device supports error correction memory access.
    ONDEMANDPAGING = ZE_BIT(3)                                              ## Device supports on-demand page-faulting.

class ze_device_property_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Device properties queried using ::zeDeviceGetProperties
class ze_device_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("type", ze_device_type_t),                                     ## [out] generic device type
        ("vendorId", c_ulong),                                          ## [out] vendor id from PCI configuration
        ("deviceId", c_ulong),                                          ## [out] device id from PCI configuration.
                                                                        ## Note, the device id uses little-endian format.
        ("flags", ze_device_property_flags_t),                          ## [out] 0 (none) or a valid combination of ::ze_device_property_flag_t
        ("subdeviceId", c_ulong),                                       ## [out] sub-device id. Only valid if ::ZE_DEVICE_PROPERTY_FLAG_SUBDEVICE
                                                                        ## is set.
        ("coreClockRate", c_ulong),                                     ## [out] Clock rate for device core.
        ("maxMemAllocSize", c_ulonglong),                               ## [out] Maximum memory allocation size.
        ("maxHardwareContexts", c_ulong),                               ## [out] Maximum number of logical hardware contexts.
        ("maxCommandQueuePriority", c_ulong),                           ## [out] Maximum priority for command queues. Higher value is higher
                                                                        ## priority.
        ("numThreadsPerEU", c_ulong),                                   ## [out] Maximum number of threads per EU.
        ("physicalEUSimdWidth", c_ulong),                               ## [out] The physical EU simd width.
        ("numEUsPerSubslice", c_ulong),                                 ## [out] Maximum number of EUs per sub-slice.
        ("numSubslicesPerSlice", c_ulong),                              ## [out] Maximum number of sub-slices per slice.
        ("numSlices", c_ulong),                                         ## [out] Maximum number of slices.
        ("timerResolution", c_ulonglong),                               ## [out] Returns the resolution of device timer used for profiling,
                                                                        ## timestamps, etc. When stype==::ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES the
                                                                        ## units are in nanoseconds. When
                                                                        ## stype==::ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2 units are in
                                                                        ## cycles/sec
        ("timestampValidBits", c_ulong),                                ## [out] Returns the number of valid bits in the timestamp value.
        ("kernelTimestampValidBits", c_ulong),                          ## [out] Returns the number of valid bits in the kernel timestamp values
        ("uuid", ze_device_uuid_t),                                     ## [out] universal unique identifier. Note: Subdevices will have their
                                                                        ## own uuid.
        ("name", c_char * ZE_MAX_DEVICE_NAME)                           ## [out] Device name
    ]

###############################################################################
## @brief Device thread identifier.
class ze_device_thread_t(Structure):
    _fields_ = [
        ("slice", c_ulong),                                             ## [in,out] the slice number.
                                                                        ## Must be `UINT32_MAX` (all) or less than the `numSlices` member of ::ze_device_properties_t.
        ("subslice", c_ulong),                                          ## [in,out] the sub-slice number within its slice.
                                                                        ## Must be `UINT32_MAX` (all) or less than the `numSubslicesPerSlice`
                                                                        ## member of ::ze_device_properties_t.
        ("eu", c_ulong),                                                ## [in,out] the EU number within its sub-slice.
                                                                        ## Must be `UINT32_MAX` (all) or less than the `numEUsPerSubslice` member
                                                                        ## of ::ze_device_properties_t.
        ("thread", c_ulong)                                             ## [in,out] the thread number within its EU.
                                                                        ## Must be `UINT32_MAX` (all) or less than the `numThreadsPerEU` member
                                                                        ## of ::ze_device_properties_t.
    ]

###############################################################################
## @brief Maximum number of subgroup sizes supported.
ZE_SUBGROUPSIZE_COUNT = 8

###############################################################################
## @brief Device compute properties queried using ::zeDeviceGetComputeProperties
class ze_device_compute_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("maxTotalGroupSize", c_ulong),                                 ## [out] Maximum items per compute group. (groupSizeX * groupSizeY *
                                                                        ## groupSizeZ) <= maxTotalGroupSize
        ("maxGroupSizeX", c_ulong),                                     ## [out] Maximum items for X dimension in group
        ("maxGroupSizeY", c_ulong),                                     ## [out] Maximum items for Y dimension in group
        ("maxGroupSizeZ", c_ulong),                                     ## [out] Maximum items for Z dimension in group
        ("maxGroupCountX", c_ulong),                                    ## [out] Maximum groups that can be launched for x dimension
        ("maxGroupCountY", c_ulong),                                    ## [out] Maximum groups that can be launched for y dimension
        ("maxGroupCountZ", c_ulong),                                    ## [out] Maximum groups that can be launched for z dimension
        ("maxSharedLocalMemory", c_ulong),                              ## [out] Maximum shared local memory per group.
        ("numSubGroupSizes", c_ulong),                                  ## [out] Number of subgroup sizes supported. This indicates number of
                                                                        ## entries in subGroupSizes.
        ("subGroupSizes", c_ulong * ZE_SUBGROUPSIZE_COUNT)              ## [out] Size group sizes supported.
    ]

###############################################################################
## @brief Maximum native kernel universal unique id (UUID) size in bytes
ZE_MAX_NATIVE_KERNEL_UUID_SIZE = 16

###############################################################################
## @brief Native kernel universal unique id (UUID)
class ze_native_kernel_uuid_t(Structure):
    _fields_ = [
        ("id", c_ubyte * ZE_MAX_NATIVE_KERNEL_UUID_SIZE)                ## [out] opaque data representing a native kernel UUID
    ]

###############################################################################
## @brief Supported device module flags
class ze_device_module_flags_v(IntEnum):
    FP16 = ZE_BIT(0)                                                        ## Device supports 16-bit floating-point operations
    FP64 = ZE_BIT(1)                                                        ## Device supports 64-bit floating-point operations
    INT64_ATOMICS = ZE_BIT(2)                                               ## Device supports 64-bit atomic operations
    DP4A = ZE_BIT(3)                                                        ## Device supports four component dot product and accumulate operations

class ze_device_module_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Supported floating-Point capability flags
class ze_device_fp_flags_v(IntEnum):
    DENORM = ZE_BIT(0)                                                      ## Supports denorms
    INF_NAN = ZE_BIT(1)                                                     ## Supports INF and quiet NaNs
    ROUND_TO_NEAREST = ZE_BIT(2)                                            ## Supports rounding to nearest even rounding mode
    ROUND_TO_ZERO = ZE_BIT(3)                                               ## Supports rounding to zero.
    ROUND_TO_INF = ZE_BIT(4)                                                ## Supports rounding to both positive and negative INF.
    FMA = ZE_BIT(5)                                                         ## Supports IEEE754-2008 fused multiply-add.
    ROUNDED_DIVIDE_SQRT = ZE_BIT(6)                                         ## Supports rounding as defined by IEEE754 for divide and sqrt
                                                                            ## operations.
    SOFT_FLOAT = ZE_BIT(7)                                                  ## Uses software implementation for basic floating-point operations.

class ze_device_fp_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Device module properties queried using ::zeDeviceGetModuleProperties
class ze_device_module_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("spirvVersionSupported", c_ulong),                             ## [out] Maximum supported SPIR-V version.
                                                                        ## Returns zero if SPIR-V is not supported.
                                                                        ## Contains major and minor attributes, use ::ZE_MAJOR_VERSION and ::ZE_MINOR_VERSION.
        ("flags", ze_device_module_flags_t),                            ## [out] 0 or a valid combination of ::ze_device_module_flag_t
        ("fp16flags", ze_device_fp_flags_t),                            ## [out] Capabilities for half-precision floating-point operations.
                                                                        ## returns 0 (if ::ZE_DEVICE_MODULE_FLAG_FP16 is not set) or a
                                                                        ## combination of ::ze_device_fp_flag_t.
        ("fp32flags", ze_device_fp_flags_t),                            ## [out] Capabilities for single-precision floating-point operations.
                                                                        ## returns a combination of ::ze_device_fp_flag_t.
        ("fp64flags", ze_device_fp_flags_t),                            ## [out] Capabilities for double-precision floating-point operations.
                                                                        ## returns 0 (if ::ZE_DEVICE_MODULE_FLAG_FP64 is not set) or a
                                                                        ## combination of ::ze_device_fp_flag_t.
        ("maxArgumentsSize", c_ulong),                                  ## [out] Maximum kernel argument size that is supported.
        ("printfBufferSize", c_ulong),                                  ## [out] Maximum size of internal buffer that holds output of printf
                                                                        ## calls from kernel.
        ("nativeKernelSupported", ze_native_kernel_uuid_t)              ## [out] Compatibility UUID of supported native kernel.
                                                                        ## UUID may or may not be the same across driver release, devices, or
                                                                        ## operating systems.
                                                                        ## Application is responsible for ensuring UUID matches before creating
                                                                        ## module using
                                                                        ## previously created native kernel.
    ]

###############################################################################
## @brief Supported command queue group property flags
class ze_command_queue_group_property_flags_v(IntEnum):
    COMPUTE = ZE_BIT(0)                                                     ## Command queue group supports enqueing compute commands.
    COPY = ZE_BIT(1)                                                        ## Command queue group supports enqueing copy commands.
    COOPERATIVE_KERNELS = ZE_BIT(2)                                         ## Command queue group supports cooperative kernels.
                                                                            ## See ::zeCommandListAppendLaunchCooperativeKernel for more details.
    METRICS = ZE_BIT(3)                                                     ## Command queue groups supports metric queries.

class ze_command_queue_group_property_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Command queue group properties queried using
##        ::zeDeviceGetCommandQueueGroupProperties
class ze_command_queue_group_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_command_queue_group_property_flags_t),             ## [out] 0 (none) or a valid combination of
                                                                        ## ::ze_command_queue_group_property_flag_t
        ("maxMemoryFillPatternSize", c_size_t),                         ## [out] maximum `pattern_size` supported by command queue group.
                                                                        ## See ::zeCommandListAppendMemoryFill for more details.
        ("numQueues", c_ulong)                                          ## [out] the number of physical engines within the group.
    ]

###############################################################################
## @brief Supported device memory property flags
class ze_device_memory_property_flags_v(IntEnum):
    TBD = ZE_BIT(0)                                                         ## reserved for future use

class ze_device_memory_property_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Device local memory properties queried using
##        ::zeDeviceGetMemoryProperties
class ze_device_memory_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_device_memory_property_flags_t),                   ## [out] 0 (none) or a valid combination of
                                                                        ## ::ze_device_memory_property_flag_t
        ("maxClockRate", c_ulong),                                      ## [out] Maximum clock rate for device memory.
        ("maxBusWidth", c_ulong),                                       ## [out] Maximum bus width between device and memory.
        ("totalSize", c_ulonglong),                                     ## [out] Total memory size in bytes that is available to the device.
        ("name", c_char * ZE_MAX_DEVICE_NAME)                           ## [out] Memory name
    ]

###############################################################################
## @brief Memory access capability flags
## 
## @details
##     - Supported access capabilities for different types of memory
##       allocations
class ze_memory_access_cap_flags_v(IntEnum):
    RW = ZE_BIT(0)                                                          ## Supports load/store access
    ATOMIC = ZE_BIT(1)                                                      ## Supports atomic access
    CONCURRENT = ZE_BIT(2)                                                  ## Supports concurrent access
    CONCURRENT_ATOMIC = ZE_BIT(3)                                           ## Supports concurrent atomic access

class ze_memory_access_cap_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Device memory access properties queried using
##        ::zeDeviceGetMemoryAccessProperties
class ze_device_memory_access_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("hostAllocCapabilities", ze_memory_access_cap_flags_t),        ## [out] host memory capabilities.
                                                                        ## returns 0 (unsupported) or a combination of ::ze_memory_access_cap_flag_t.
        ("deviceAllocCapabilities", ze_memory_access_cap_flags_t),      ## [out] device memory capabilities.
                                                                        ## returns 0 (unsupported) or a combination of ::ze_memory_access_cap_flag_t.
        ("sharedSingleDeviceAllocCapabilities", ze_memory_access_cap_flags_t),  ## [out] shared, single-device memory capabilities.
                                                                        ## returns 0 (unsupported) or a combination of ::ze_memory_access_cap_flag_t.
        ("sharedCrossDeviceAllocCapabilities", ze_memory_access_cap_flags_t),   ## [out] shared, cross-device memory capabilities.
                                                                        ## returns 0 (unsupported) or a combination of ::ze_memory_access_cap_flag_t.
        ("sharedSystemAllocCapabilities", ze_memory_access_cap_flags_t) ## [out] shared, system memory capabilities.
                                                                        ## returns 0 (unsupported) or a combination of ::ze_memory_access_cap_flag_t.
    ]

###############################################################################
## @brief Supported cache control property flags
class ze_device_cache_property_flags_v(IntEnum):
    USER_CONTROL = ZE_BIT(0)                                                ## Device support User Cache Control (i.e. SLM section vs Generic Cache)

class ze_device_cache_property_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Device cache properties queried using ::zeDeviceGetCacheProperties
class ze_device_cache_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_device_cache_property_flags_t),                    ## [out] 0 (none) or a valid combination of
                                                                        ## ::ze_device_cache_property_flag_t
        ("cacheSize", c_size_t)                                         ## [out] Per-cache size, in bytes
    ]

###############################################################################
## @brief Device image properties queried using ::zeDeviceGetImageProperties
class ze_device_image_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("maxImageDims1D", c_ulong),                                    ## [out] Maximum image dimensions for 1D resources. if 0, then 1D images
                                                                        ## are unsupported.
        ("maxImageDims2D", c_ulong),                                    ## [out] Maximum image dimensions for 2D resources. if 0, then 2D images
                                                                        ## are unsupported.
        ("maxImageDims3D", c_ulong),                                    ## [out] Maximum image dimensions for 3D resources. if 0, then 3D images
                                                                        ## are unsupported.
        ("maxImageBufferSize", c_ulonglong),                            ## [out] Maximum image buffer size in bytes. if 0, then buffer images are
                                                                        ## unsupported.
        ("maxImageArraySlices", c_ulong),                               ## [out] Maximum image array slices. if 0, then image arrays are
                                                                        ## unsupported.
        ("maxSamplers", c_ulong),                                       ## [out] Max samplers that can be used in kernel. if 0, then sampling is
                                                                        ## unsupported.
        ("maxReadImageArgs", c_ulong),                                  ## [out] Returns the maximum number of simultaneous image objects that
                                                                        ## can be read from by a kernel. if 0, then reading images is
                                                                        ## unsupported.
        ("maxWriteImageArgs", c_ulong)                                  ## [out] Returns the maximum number of simultaneous image objects that
                                                                        ## can be written to by a kernel. if 0, then writing images is
                                                                        ## unsupported.
    ]

###############################################################################
## @brief Device external memory import and export properties
class ze_device_external_memory_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("memoryAllocationImportTypes", ze_external_memory_type_flags_t),   ## [out] Supported external memory import types for memory allocations.
        ("memoryAllocationExportTypes", ze_external_memory_type_flags_t),   ## [out] Supported external memory export types for memory allocations.
        ("imageImportTypes", ze_external_memory_type_flags_t),          ## [out] Supported external memory import types for images.
        ("imageExportTypes", ze_external_memory_type_flags_t)           ## [out] Supported external memory export types for images.
    ]

###############################################################################
## @brief Supported device peer-to-peer property flags
class ze_device_p2p_property_flags_v(IntEnum):
    ACCESS = ZE_BIT(0)                                                      ## Device supports access between peer devices.
    ATOMICS = ZE_BIT(1)                                                     ## Device supports atomics between peer devices.

class ze_device_p2p_property_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Device peer-to-peer properties queried using
##        ::zeDeviceGetP2PProperties
class ze_device_p2p_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_device_p2p_property_flags_t)                       ## [out] 0 (none) or a valid combination of
                                                                        ## ::ze_device_p2p_property_flag_t
    ]

###############################################################################
## @brief Supported context creation flags
class ze_context_flags_v(IntEnum):
    TBD = ZE_BIT(0)                                                         ## reserved for future use

class ze_context_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Context descriptor
class ze_context_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_context_flags_t)                                   ## [in] creation flags.
                                                                        ## must be 0 (default) or a valid combination of ::ze_context_flag_t;
                                                                        ## default behavior may use implicit driver-based heuristics.
    ]

###############################################################################
## @brief Supported command queue flags
class ze_command_queue_flags_v(IntEnum):
    EXPLICIT_ONLY = ZE_BIT(0)                                               ## command queue should be optimized for submission to a single device engine.
                                                                            ## driver **must** disable any implicit optimizations for distributing
                                                                            ## work across multiple engines.
                                                                            ## this flag should be used when applications want full control over
                                                                            ## multi-engine submission and scheduling.
    IN_ORDER = ZE_BIT(1)                                                    ## To be used only when creating immediate command lists. Commands
                                                                            ## appended to the immediate command
                                                                            ## list are executed in-order, with driver implementation enforcing
                                                                            ## dependencies between them.
                                                                            ## Application is not required to have the signal event of a given
                                                                            ## command being the wait event of
                                                                            ## the next to define an in-order list, and application is allowed to
                                                                            ## pass signal and wait events
                                                                            ## to each appended command to implement more complex dependency graphs.

class ze_command_queue_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Supported command queue modes
class ze_command_queue_mode_v(IntEnum):
    DEFAULT = 0                                                             ## implicit default behavior; uses driver-based heuristics
    SYNCHRONOUS = 1                                                         ## Device execution always completes immediately on execute;
                                                                            ## Host thread is blocked using wait on implicit synchronization object
    ASYNCHRONOUS = 2                                                        ## Device execution is scheduled and will complete in future;
                                                                            ## explicit synchronization object must be used to determine completeness

class ze_command_queue_mode_t(c_int):
    def __str__(self):
        return str(ze_command_queue_mode_v(self.value))


###############################################################################
## @brief Supported command queue priorities
class ze_command_queue_priority_v(IntEnum):
    NORMAL = 0                                                              ## [default] normal priority
    PRIORITY_LOW = 1                                                        ## lower priority than normal
    PRIORITY_HIGH = 2                                                       ## higher priority than normal

class ze_command_queue_priority_t(c_int):
    def __str__(self):
        return str(ze_command_queue_priority_v(self.value))


###############################################################################
## @brief Command Queue descriptor
class ze_command_queue_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("ordinal", c_ulong),                                           ## [in] command queue group ordinal
        ("index", c_ulong),                                             ## [in] command queue index within the group;
                                                                        ## must be zero if ::ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY is not set
        ("flags", ze_command_queue_flags_t),                            ## [in] usage flags.
                                                                        ## must be 0 (default) or a valid combination of ::ze_command_queue_flag_t;
                                                                        ## default behavior may use implicit driver-based heuristics to balance
                                                                        ## latency and throughput.
        ("mode", ze_command_queue_mode_t),                              ## [in] operation mode
        ("priority", ze_command_queue_priority_t)                       ## [in] priority
    ]

###############################################################################
## @brief Supported command list creation flags
class ze_command_list_flags_v(IntEnum):
    RELAXED_ORDERING = ZE_BIT(0)                                            ## driver may reorder commands (e.g., kernels, copies) between barriers
                                                                            ## and synchronization primitives.
                                                                            ## using this flag may increase Host overhead of ::zeCommandListClose.
                                                                            ## therefore, this flag should **not** be set for low-latency usage-models.
    MAXIMIZE_THROUGHPUT = ZE_BIT(1)                                         ## driver may perform additional optimizations that increase execution
                                                                            ## throughput. 
                                                                            ## using this flag may increase Host overhead of ::zeCommandListClose and ::zeCommandQueueExecuteCommandLists.
                                                                            ## therefore, this flag should **not** be set for low-latency usage-models.
    EXPLICIT_ONLY = ZE_BIT(2)                                               ## command list should be optimized for submission to a single command
                                                                            ## queue and device engine.
                                                                            ## driver **must** disable any implicit optimizations for distributing
                                                                            ## work across multiple engines.
                                                                            ## this flag should be used when applications want full control over
                                                                            ## multi-engine submission and scheduling.
    IN_ORDER = ZE_BIT(3)                                                    ## commands appended to this command list are executed in-order, with
                                                                            ## driver implementation
                                                                            ## enforcing dependencies between them. Application is not required to
                                                                            ## have the signal event
                                                                            ## of a given command being the wait event of the next to define an
                                                                            ## in-order list, and
                                                                            ## application is allowed to pass signal and wait events to each appended
                                                                            ## command to implement
                                                                            ## more complex dependency graphs. Cannot be combined with ::ZE_COMMAND_LIST_FLAG_RELAXED_ORDERING.
    EXP_CLONEABLE = ZE_BIT(4)                                               ## this command list may be cloned using ::zeCommandListCreateCloneExp
                                                                            ## after ::zeCommandListClose.

class ze_command_list_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Command List descriptor
class ze_command_list_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("commandQueueGroupOrdinal", c_ulong),                          ## [in] command queue group ordinal to which this command list will be
                                                                        ## submitted
        ("flags", ze_command_list_flags_t)                              ## [in] usage flags.
                                                                        ## must be 0 (default) or a valid combination of ::ze_command_list_flag_t;
                                                                        ## default behavior may use implicit driver-based heuristics to balance
                                                                        ## latency and throughput.
    ]

###############################################################################
## @brief Copy region descriptor
class ze_copy_region_t(Structure):
    _fields_ = [
        ("originX", c_ulong),                                           ## [in] The origin x offset for region in bytes
        ("originY", c_ulong),                                           ## [in] The origin y offset for region in rows
        ("originZ", c_ulong),                                           ## [in] The origin z offset for region in slices
        ("width", c_ulong),                                             ## [in] The region width relative to origin in bytes
        ("height", c_ulong),                                            ## [in] The region height relative to origin in rows
        ("depth", c_ulong)                                              ## [in] The region depth relative to origin in slices. Set this to 0 for
                                                                        ## 2D copy.
    ]

###############################################################################
## @brief Region descriptor
class ze_image_region_t(Structure):
    _fields_ = [
        ("originX", c_ulong),                                           ## [in] The origin x offset for region in pixels
        ("originY", c_ulong),                                           ## [in] The origin y offset for region in pixels
        ("originZ", c_ulong),                                           ## [in] The origin z offset for region in pixels
        ("width", c_ulong),                                             ## [in] The region width relative to origin in pixels
        ("height", c_ulong),                                            ## [in] The region height relative to origin in pixels
        ("depth", c_ulong)                                              ## [in] The region depth relative to origin. For 1D or 2D images, set
                                                                        ## this to 1.
    ]

###############################################################################
## @brief Supported memory advice hints
class ze_memory_advice_v(IntEnum):
    SET_READ_MOSTLY = 0                                                     ## hint that memory will be read from frequently and written to rarely
    CLEAR_READ_MOSTLY = 1                                                   ## removes the effect of ::ZE_MEMORY_ADVICE_SET_READ_MOSTLY
    SET_PREFERRED_LOCATION = 2                                              ## hint that the preferred memory location is the specified device
    CLEAR_PREFERRED_LOCATION = 3                                            ## removes the effect of ::ZE_MEMORY_ADVICE_SET_PREFERRED_LOCATION
    SET_NON_ATOMIC_MOSTLY = 4                                               ## hints that memory will mostly be accessed non-atomically
    CLEAR_NON_ATOMIC_MOSTLY = 5                                             ## removes the effect of ::ZE_MEMORY_ADVICE_SET_NON_ATOMIC_MOSTLY
    BIAS_CACHED = 6                                                         ## hints that memory should be cached
    BIAS_UNCACHED = 7                                                       ## hints that memory should be not be cached
    SET_SYSTEM_MEMORY_PREFERRED_LOCATION = 8                                ## hint that the preferred memory location is host memory
    CLEAR_SYSTEM_MEMORY_PREFERRED_LOCATION = 9                              ## removes the effect of
                                                                            ## ::ZE_MEMORY_ADVICE_SET_SYSTEM_MEMORY_PREFERRED_LOCATION

class ze_memory_advice_t(c_int):
    def __str__(self):
        return str(ze_memory_advice_v(self.value))


###############################################################################
## @brief Supported event pool creation flags
class ze_event_pool_flags_v(IntEnum):
    HOST_VISIBLE = ZE_BIT(0)                                                ## signals and waits are also visible to host
    IPC = ZE_BIT(1)                                                         ## signals and waits may be shared across processes
    KERNEL_TIMESTAMP = ZE_BIT(2)                                            ## Indicates all events in pool will contain kernel timestamps
    KERNEL_MAPPED_TIMESTAMP = ZE_BIT(3)                                     ## Indicates all events in pool will contain kernel timestamps
                                                                            ## synchronized to host time domain; cannot be combined with
                                                                            ## ::ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP

class ze_event_pool_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Event pool descriptor
class ze_event_pool_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_event_pool_flags_t),                               ## [in] creation flags.
                                                                        ## must be 0 (default) or a valid combination of ::ze_event_pool_flag_t;
                                                                        ## default behavior is signals and waits are visible to the entire device
                                                                        ## and peer devices.
        ("count", c_ulong)                                              ## [in] number of events within the pool; must be greater than 0
    ]

###############################################################################
## @brief Supported event scope flags
class ze_event_scope_flags_v(IntEnum):
    SUBDEVICE = ZE_BIT(0)                                                   ## cache hierarchies are flushed or invalidated sufficient for local
                                                                            ## sub-device access
    DEVICE = ZE_BIT(1)                                                      ## cache hierarchies are flushed or invalidated sufficient for global
                                                                            ## device access and peer device access
    HOST = ZE_BIT(2)                                                        ## cache hierarchies are flushed or invalidated sufficient for device and
                                                                            ## host access

class ze_event_scope_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Event descriptor
class ze_event_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("index", c_ulong),                                             ## [in] index of the event within the pool; must be less than the count
                                                                        ## specified during pool creation
        ("signal", ze_event_scope_flags_t),                             ## [in] defines the scope of relevant cache hierarchies to flush on a
                                                                        ## signal action before the event is triggered.
                                                                        ## must be 0 (default) or a valid combination of ::ze_event_scope_flag_t;
                                                                        ## default behavior is synchronization within the command list only, no
                                                                        ## additional cache hierarchies are flushed.
        ("wait", ze_event_scope_flags_t)                                ## [in] defines the scope of relevant cache hierarchies to invalidate on
                                                                        ## a wait action after the event is complete.
                                                                        ## must be 0 (default) or a valid combination of ::ze_event_scope_flag_t;
                                                                        ## default behavior is synchronization within the command list only, no
                                                                        ## additional cache hierarchies are invalidated.
    ]

###############################################################################
## @brief Kernel timestamp clock data
## 
## @details
##     - The timestamp frequency can be queried from the `timerResolution`
##       member of ::ze_device_properties_t.
##     - The number of valid bits in the timestamp value can be queried from
##       the `kernelTimestampValidBits` member of ::ze_device_properties_t.
class ze_kernel_timestamp_data_t(Structure):
    _fields_ = [
        ("kernelStart", c_ulonglong),                                   ## [out] device clock at start of kernel execution
        ("kernelEnd", c_ulonglong)                                      ## [out] device clock at end of kernel execution
    ]

###############################################################################
## @brief Kernel timestamp result
class ze_kernel_timestamp_result_t(Structure):
    _fields_ = [
        ("global", ze_kernel_timestamp_data_t),                         ## [out] wall-clock data
        ("context", ze_kernel_timestamp_data_t)                         ## [out] context-active data; only includes clocks while device context
                                                                        ## was actively executing.
    ]

###############################################################################
## @brief Supported fence creation flags
class ze_fence_flags_v(IntEnum):
    SIGNALED = ZE_BIT(0)                                                    ## fence is created in the signaled state, otherwise not signaled.

class ze_fence_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Fence descriptor
class ze_fence_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_fence_flags_t)                                     ## [in] creation flags.
                                                                        ## must be 0 (default) or a valid combination of ::ze_fence_flag_t.
    ]

###############################################################################
## @brief Supported image creation flags
class ze_image_flags_v(IntEnum):
    KERNEL_WRITE = ZE_BIT(0)                                                ## kernels will write contents
    BIAS_UNCACHED = ZE_BIT(1)                                               ## device should not cache contents

class ze_image_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Supported image types
class ze_image_type_v(IntEnum):
    _1D = 0                                                                 ## 1D
    _1DARRAY = 1                                                            ## 1D array
    _2D = 2                                                                 ## 2D
    _2DARRAY = 3                                                            ## 2D array
    _3D = 4                                                                 ## 3D
    BUFFER = 5                                                              ## Buffer

class ze_image_type_t(c_int):
    def __str__(self):
        return str(ze_image_type_v(self.value))


###############################################################################
## @brief Supported image format layouts
class ze_image_format_layout_v(IntEnum):
    _8 = 0                                                                  ## 8-bit single component layout
    _16 = 1                                                                 ## 16-bit single component layout
    _32 = 2                                                                 ## 32-bit single component layout
    _8_8 = 3                                                                ## 2-component 8-bit layout
    _8_8_8_8 = 4                                                            ## 4-component 8-bit layout
    _16_16 = 5                                                              ## 2-component 16-bit layout
    _16_16_16_16 = 6                                                        ## 4-component 16-bit layout
    _32_32 = 7                                                              ## 2-component 32-bit layout
    _32_32_32_32 = 8                                                        ## 4-component 32-bit layout
    _10_10_10_2 = 9                                                         ## 4-component 10_10_10_2 layout
    _11_11_10 = 10                                                          ## 3-component 11_11_10 layout
    _5_6_5 = 11                                                             ## 3-component 5_6_5 layout
    _5_5_5_1 = 12                                                           ## 4-component 5_5_5_1 layout
    _4_4_4_4 = 13                                                           ## 4-component 4_4_4_4 layout
    Y8 = 14                                                                 ## Media Format: Y8. Format type and swizzle is ignored for this.
    NV12 = 15                                                               ## Media Format: NV12. Format type and swizzle is ignored for this.
    YUYV = 16                                                               ## Media Format: YUYV. Format type and swizzle is ignored for this.
    VYUY = 17                                                               ## Media Format: VYUY. Format type and swizzle is ignored for this.
    YVYU = 18                                                               ## Media Format: YVYU. Format type and swizzle is ignored for this.
    UYVY = 19                                                               ## Media Format: UYVY. Format type and swizzle is ignored for this.
    AYUV = 20                                                               ## Media Format: AYUV. Format type and swizzle is ignored for this.
    P010 = 21                                                               ## Media Format: P010. Format type and swizzle is ignored for this.
    Y410 = 22                                                               ## Media Format: Y410. Format type and swizzle is ignored for this.
    P012 = 23                                                               ## Media Format: P012. Format type and swizzle is ignored for this.
    Y16 = 24                                                                ## Media Format: Y16. Format type and swizzle is ignored for this.
    P016 = 25                                                               ## Media Format: P016. Format type and swizzle is ignored for this.
    Y216 = 26                                                               ## Media Format: Y216. Format type and swizzle is ignored for this.
    P216 = 27                                                               ## Media Format: P216. Format type and swizzle is ignored for this.
    P8 = 28                                                                 ## Media Format: P8. Format type and swizzle is ignored for this.
    YUY2 = 29                                                               ## Media Format: YUY2. Format type and swizzle is ignored for this.
    A8P8 = 30                                                               ## Media Format: A8P8. Format type and swizzle is ignored for this.
    IA44 = 31                                                               ## Media Format: IA44. Format type and swizzle is ignored for this.
    AI44 = 32                                                               ## Media Format: AI44. Format type and swizzle is ignored for this.
    Y416 = 33                                                               ## Media Format: Y416. Format type and swizzle is ignored for this.
    Y210 = 34                                                               ## Media Format: Y210. Format type and swizzle is ignored for this.
    I420 = 35                                                               ## Media Format: I420. Format type and swizzle is ignored for this.
    YV12 = 36                                                               ## Media Format: YV12. Format type and swizzle is ignored for this.
    _400P = 37                                                              ## Media Format: 400P. Format type and swizzle is ignored for this.
    _422H = 38                                                              ## Media Format: 422H. Format type and swizzle is ignored for this.
    _422V = 39                                                              ## Media Format: 422V. Format type and swizzle is ignored for this.
    _444P = 40                                                              ## Media Format: 444P. Format type and swizzle is ignored for this.
    RGBP = 41                                                               ## Media Format: RGBP. Format type and swizzle is ignored for this.
    BRGP = 42                                                               ## Media Format: BRGP. Format type and swizzle is ignored for this.
    _8_8_8 = 43                                                             ## 3-component 8-bit layout
    _16_16_16 = 44                                                          ## 3-component 16-bit layout
    _32_32_32 = 45                                                          ## 3-component 32-bit layout

class ze_image_format_layout_t(c_int):
    def __str__(self):
        return str(ze_image_format_layout_v(self.value))


###############################################################################
## @brief Supported image format types
class ze_image_format_type_v(IntEnum):
    UINT = 0                                                                ## Unsigned integer
    SINT = 1                                                                ## Signed integer
    UNORM = 2                                                               ## Unsigned normalized integer
    SNORM = 3                                                               ## Signed normalized integer
    FLOAT = 4                                                               ## Float

class ze_image_format_type_t(c_int):
    def __str__(self):
        return str(ze_image_format_type_v(self.value))


###############################################################################
## @brief Supported image format component swizzle into channel
class ze_image_format_swizzle_v(IntEnum):
    R = 0                                                                   ## Red component
    G = 1                                                                   ## Green component
    B = 2                                                                   ## Blue component
    A = 3                                                                   ## Alpha component
    _0 = 4                                                                  ## Zero
    _1 = 5                                                                  ## One
    X = 6                                                                   ## Don't care

class ze_image_format_swizzle_t(c_int):
    def __str__(self):
        return str(ze_image_format_swizzle_v(self.value))


###############################################################################
## @brief Image format 
class ze_image_format_t(Structure):
    _fields_ = [
        ("layout", ze_image_format_layout_t),                           ## [in] image format component layout (e.g. N-component layouts and media
                                                                        ## formats)
        ("type", ze_image_format_type_t),                               ## [in] image format type
        ("x", ze_image_format_swizzle_t),                               ## [in] image component swizzle into channel x
        ("y", ze_image_format_swizzle_t),                               ## [in] image component swizzle into channel y
        ("z", ze_image_format_swizzle_t),                               ## [in] image component swizzle into channel z
        ("w", ze_image_format_swizzle_t)                                ## [in] image component swizzle into channel w
    ]

###############################################################################
## @brief Image descriptor
class ze_image_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_image_flags_t),                                    ## [in] creation flags.
                                                                        ## must be 0 (default) or a valid combination of ::ze_image_flag_t;
                                                                        ## default is read-only, cached access.
        ("type", ze_image_type_t),                                      ## [in] image type. Media format layouts are unsupported for
                                                                        ## ::ZE_IMAGE_TYPE_BUFFER
        ("format", ze_image_format_t),                                  ## [in] image format
        ("width", c_ulonglong),                                         ## [in] width dimension.
                                                                        ## ::ZE_IMAGE_TYPE_BUFFER: size in bytes; see the `maxImageBufferSize`
                                                                        ## member of ::ze_device_image_properties_t for limits.
                                                                        ## ::ZE_IMAGE_TYPE_1D, ::ZE_IMAGE_TYPE_1DARRAY: width in pixels; see the
                                                                        ## `maxImageDims1D` member of ::ze_device_image_properties_t for limits.
                                                                        ## ::ZE_IMAGE_TYPE_2D, ::ZE_IMAGE_TYPE_2DARRAY: width in pixels; see the
                                                                        ## `maxImageDims2D` member of ::ze_device_image_properties_t for limits.
                                                                        ## ::ZE_IMAGE_TYPE_3D: width in pixels; see the `maxImageDims3D` member
                                                                        ## of ::ze_device_image_properties_t for limits.
        ("height", c_ulong),                                            ## [in] height dimension.
                                                                        ## ::ZE_IMAGE_TYPE_2D, ::ZE_IMAGE_TYPE_2DARRAY: height in pixels; see the
                                                                        ## `maxImageDims2D` member of ::ze_device_image_properties_t for limits.
                                                                        ## ::ZE_IMAGE_TYPE_3D: height in pixels; see the `maxImageDims3D` member
                                                                        ## of ::ze_device_image_properties_t for limits.
                                                                        ## other: ignored.
        ("depth", c_ulong),                                             ## [in] depth dimension.
                                                                        ## ::ZE_IMAGE_TYPE_3D: depth in pixels; see the `maxImageDims3D` member
                                                                        ## of ::ze_device_image_properties_t for limits.
                                                                        ## other: ignored.
        ("arraylevels", c_ulong),                                       ## [in] array levels.
                                                                        ## ::ZE_IMAGE_TYPE_1DARRAY, ::ZE_IMAGE_TYPE_2DARRAY: see the
                                                                        ## `maxImageArraySlices` member of ::ze_device_image_properties_t for limits.
                                                                        ## other: ignored.
        ("miplevels", c_ulong)                                          ## [in] mipmap levels (must be 0)
    ]

###############################################################################
## @brief Supported sampler filtering flags
class ze_image_sampler_filter_flags_v(IntEnum):
    POINT = ZE_BIT(0)                                                       ## device supports point filtering
    LINEAR = ZE_BIT(1)                                                      ## device supports linear filtering

class ze_image_sampler_filter_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Image properties
class ze_image_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("samplerFilterFlags", ze_image_sampler_filter_flags_t)         ## [out] supported sampler filtering.
                                                                        ## returns 0 (unsupported) or a combination of ::ze_image_sampler_filter_flag_t.
    ]

###############################################################################
## @brief Supported memory allocation flags
class ze_device_mem_alloc_flags_v(IntEnum):
    BIAS_CACHED = ZE_BIT(0)                                                 ## device should cache allocation
    BIAS_UNCACHED = ZE_BIT(1)                                               ## device should not cache allocation (UC)
    BIAS_INITIAL_PLACEMENT = ZE_BIT(2)                                      ## optimize shared allocation for first access on the device

class ze_device_mem_alloc_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Device memory allocation descriptor
class ze_device_mem_alloc_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_device_mem_alloc_flags_t),                         ## [in] flags specifying additional allocation controls.
                                                                        ## must be 0 (default) or a valid combination of ::ze_device_mem_alloc_flag_t;
                                                                        ## default behavior may use implicit driver-based heuristics.
        ("ordinal", c_ulong)                                            ## [in] ordinal of the device's local memory to allocate from.
                                                                        ## must be less than the count returned from ::zeDeviceGetMemoryProperties.
    ]

###############################################################################
## @brief Supported host memory allocation flags
class ze_host_mem_alloc_flags_v(IntEnum):
    BIAS_CACHED = ZE_BIT(0)                                                 ## host should cache allocation
    BIAS_UNCACHED = ZE_BIT(1)                                               ## host should not cache allocation (UC)
    BIAS_WRITE_COMBINED = ZE_BIT(2)                                         ## host memory should be allocated write-combined (WC)
    BIAS_INITIAL_PLACEMENT = ZE_BIT(3)                                      ## optimize shared allocation for first access on the host

class ze_host_mem_alloc_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Host memory allocation descriptor
class ze_host_mem_alloc_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_host_mem_alloc_flags_t)                            ## [in] flags specifying additional allocation controls.
                                                                        ## must be 0 (default) or a valid combination of ::ze_host_mem_alloc_flag_t;
                                                                        ## default behavior may use implicit driver-based heuristics.
    ]

###############################################################################
## @brief Memory allocation type
class ze_memory_type_v(IntEnum):
    UNKNOWN = 0                                                             ## the memory pointed to is of unknown type
    HOST = 1                                                                ## the memory pointed to is a host allocation
    DEVICE = 2                                                              ## the memory pointed to is a device allocation
    SHARED = 3                                                              ## the memory pointed to is a shared ownership allocation

class ze_memory_type_t(c_int):
    def __str__(self):
        return str(ze_memory_type_v(self.value))


###############################################################################
## @brief Memory allocation properties queried using ::zeMemGetAllocProperties
class ze_memory_allocation_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("type", ze_memory_type_t),                                     ## [out] type of allocated memory
        ("id", c_ulonglong),                                            ## [out] identifier for this allocation
        ("pageSize", c_ulonglong)                                       ## [out] page size used for allocation
    ]

###############################################################################
## @brief Supported IPC memory flags
class ze_ipc_memory_flags_v(IntEnum):
    BIAS_CACHED = ZE_BIT(0)                                                 ## device should cache allocation
    BIAS_UNCACHED = ZE_BIT(1)                                               ## device should not cache allocation (UC)

class ze_ipc_memory_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Additional allocation descriptor for exporting external memory
## 
## @details
##     - This structure may be passed to ::zeMemAllocDevice and
##       ::zeMemAllocHost, via the `pNext` member of
##       ::ze_device_mem_alloc_desc_t or ::ze_host_mem_alloc_desc_t,
##       respectively, to indicate an exportable memory allocation.
##     - This structure may be passed to ::zeImageCreate, via the `pNext`
##       member of ::ze_image_desc_t, to indicate an exportable image.
class ze_external_memory_export_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_external_memory_type_flags_t)                      ## [in] flags specifying memory export types for this allocation.
                                                                        ## must be 0 (default) or a valid combination of ::ze_external_memory_type_flags_t
    ]

###############################################################################
## @brief Additional allocation descriptor for importing external memory as a
##        file descriptor
## 
## @details
##     - This structure may be passed to ::zeMemAllocDevice or
##       ::zeMemAllocHost, via the `pNext` member of
##       ::ze_device_mem_alloc_desc_t or of ::ze_host_mem_alloc_desc_t,
##       respectively, to import memory from a file descriptor.
##     - This structure may be passed to ::zeImageCreate, via the `pNext`
##       member of ::ze_image_desc_t, to import memory from a file descriptor.
class ze_external_memory_import_fd_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_external_memory_type_flags_t),                     ## [in] flags specifying the memory import type for the file descriptor.
                                                                        ## must be 0 (default) or a valid combination of ::ze_external_memory_type_flags_t
        ("fd", c_int)                                                   ## [in] the file descriptor handle to import
    ]

###############################################################################
## @brief Exports an allocation as a file descriptor
## 
## @details
##     - This structure may be passed to ::zeMemGetAllocProperties, via the
##       `pNext` member of ::ze_memory_allocation_properties_t, to export a
##       memory allocation as a file descriptor.
##     - This structure may be passed to ::zeImageGetAllocPropertiesExt, via
##       the `pNext` member of ::ze_image_allocation_ext_properties_t, to
##       export an image as a file descriptor.
##     - The requested memory export type must have been specified when the
##       allocation was made.
class ze_external_memory_export_fd_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_external_memory_type_flags_t),                     ## [in] flags specifying the memory export type for the file descriptor.
                                                                        ## must be 0 (default) or a valid combination of ::ze_external_memory_type_flags_t
        ("fd", c_int)                                                   ## [out] the exported file descriptor handle representing the allocation.
    ]

###############################################################################
## @brief Additional allocation descriptor for importing external memory as a
##        Win32 handle
## 
## @details
##     - When `handle` is `nullptr`, `name` must not be `nullptr`.
##     - When `name` is `nullptr`, `handle` must not be `nullptr`.
##     - When `flags` is ::ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32_KMT,
##       `name` must be `nullptr`.
##     - This structure may be passed to ::zeMemAllocDevice or
##       ::zeMemAllocHost, via the `pNext` member of
##       ::ze_device_mem_alloc_desc_t or of ::ze_host_mem_alloc_desc_t,
##       respectively, to import memory from a Win32 handle.
##     - This structure may be passed to ::zeImageCreate, via the `pNext`
##       member of ::ze_image_desc_t, to import memory from a Win32 handle.
class ze_external_memory_import_win32_handle_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_external_memory_type_flags_t),                     ## [in] flags specifying the memory import type for the Win32 handle.
                                                                        ## must be 0 (default) or a valid combination of ::ze_external_memory_type_flags_t
        ("handle", c_void_p),                                           ## [in][optional] the Win32 handle to import
        ("name", c_void_p)                                              ## [in][optional] name of a memory object to import
    ]

###############################################################################
## @brief Exports an allocation as a Win32 handle
## 
## @details
##     - This structure may be passed to ::zeMemGetAllocProperties, via the
##       `pNext` member of ::ze_memory_allocation_properties_t, to export a
##       memory allocation as a Win32 handle.
##     - This structure may be passed to ::zeImageGetAllocPropertiesExt, via
##       the `pNext` member of ::ze_image_allocation_ext_properties_t, to
##       export an image as a Win32 handle.
##     - The requested memory export type must have been specified when the
##       allocation was made.
class ze_external_memory_export_win32_handle_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_external_memory_type_flags_t),                     ## [in] flags specifying the memory export type for the Win32 handle.
                                                                        ## must be 0 (default) or a valid combination of ::ze_external_memory_type_flags_t
        ("handle", c_void_p)                                            ## [out] the exported Win32 handle representing the allocation.
    ]

###############################################################################
## @brief atomic access attribute flags
class ze_memory_atomic_attr_exp_flags_v(IntEnum):
    NO_ATOMICS = ZE_BIT(0)                                                  ## Atomics on the pointer are not allowed
    NO_HOST_ATOMICS = ZE_BIT(1)                                             ## Host atomics on the pointer are not allowed
    HOST_ATOMICS = ZE_BIT(2)                                                ## Host atomics on the pointer are allowed. Requires
                                                                            ## ::ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC returned by
                                                                            ## ::zeDeviceGetMemoryAccessProperties.
    NO_DEVICE_ATOMICS = ZE_BIT(3)                                           ## Device atomics on the pointer are not allowed
    DEVICE_ATOMICS = ZE_BIT(4)                                              ## Device atomics on the pointer are allowed. Requires
                                                                            ## ::ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC returned by
                                                                            ## ::zeDeviceGetMemoryAccessProperties.
    NO_SYSTEM_ATOMICS = ZE_BIT(5)                                           ## Concurrent atomics on the pointer from both host and device are not
                                                                            ## allowed
    SYSTEM_ATOMICS = ZE_BIT(6)                                              ## Concurrent atomics on the pointer from both host and device are
                                                                            ## allowed. Requires ::ZE_MEMORY_ACCESS_CAP_FLAG_CONCURRENT_ATOMIC
                                                                            ## returned by ::zeDeviceGetMemoryAccessProperties.

class ze_memory_atomic_attr_exp_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Supported module creation input formats
class ze_module_format_v(IntEnum):
    IL_SPIRV = 0                                                            ## Format is SPIRV IL format
    NATIVE = 1                                                              ## Format is device native format

class ze_module_format_t(c_int):
    def __str__(self):
        return str(ze_module_format_v(self.value))


###############################################################################
## @brief Specialization constants - User defined constants
class ze_module_constants_t(Structure):
    _fields_ = [
        ("numConstants", c_ulong),                                      ## [in] Number of specialization constants.
        ("pConstantIds", POINTER(c_ulong)),                             ## [in][range(0, numConstants)] Array of IDs that is sized to
                                                                        ## numConstants.
        ("pConstantValues", POINTER(c_void_p))                          ## [in][range(0, numConstants)] Array of pointers to values that is sized
                                                                        ## to numConstants.
    ]

###############################################################################
## @brief Module descriptor
class ze_module_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("format", ze_module_format_t),                                 ## [in] Module format passed in with pInputModule
        ("inputSize", c_size_t),                                        ## [in] size of input IL or ISA from pInputModule.
        ("pInputModule", POINTER(c_ubyte)),                             ## [in] pointer to IL or ISA
        ("pBuildFlags", c_char_p),                                      ## [in][optional] string containing one or more (comma-separated)
                                                                        ## compiler flags. If unsupported, flag is ignored with a warning.
                                                                        ##  - "-ze-opt-disable"
                                                                        ##       - Disable optimizations
                                                                        ##  - "-ze-opt-level"
                                                                        ##       - Specifies optimization level for compiler. Levels are
                                                                        ## implementation specific.
                                                                        ##           - 0 is no optimizations (equivalent to -ze-opt-disable)
                                                                        ##           - 1 is optimize minimally (may be the same as 2)
                                                                        ##           - 2 is optimize more (default)
                                                                        ##  - "-ze-opt-greater-than-4GB-buffer-required"
                                                                        ##       - Use 64-bit offset calculations for buffers.
                                                                        ##  - "-ze-opt-large-register-file"
                                                                        ##       - Increase number of registers available to threads.
                                                                        ##  - "-ze-opt-has-buffer-offset-arg"
                                                                        ##       - Extend stateless to stateful optimization to more
                                                                        ##         cases with the use of additional offset (e.g. 64-bit
                                                                        ##         pointer to binding table with 32-bit offset).
                                                                        ##  - "-g"
                                                                        ##       - Include debugging information.
        ("pConstants", POINTER(ze_module_constants_t))                  ## [in][optional] pointer to specialization constants. Valid only for
                                                                        ## SPIR-V input. This must be set to nullptr if no specialization
                                                                        ## constants are provided.
    ]

###############################################################################
## @brief Supported module property flags
class ze_module_property_flags_v(IntEnum):
    IMPORTS = ZE_BIT(0)                                                     ## Module has imports (i.e. imported global variables and/or kernels).
                                                                            ## See ::zeModuleDynamicLink.

class ze_module_property_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Module properties
class ze_module_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_module_property_flags_t)                           ## [out] 0 (none) or a valid combination of ::ze_module_property_flag_t
    ]

###############################################################################
## @brief Supported kernel creation flags
class ze_kernel_flags_v(IntEnum):
    FORCE_RESIDENCY = ZE_BIT(0)                                             ## force all device allocations to be resident during execution
    EXPLICIT_RESIDENCY = ZE_BIT(1)                                          ## application is responsible for all residency of device allocations.
                                                                            ## driver may disable implicit residency management.

class ze_kernel_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Kernel descriptor
class ze_kernel_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_kernel_flags_t),                                   ## [in] creation flags.
                                                                        ## must be 0 (default) or a valid combination of ::ze_kernel_flag_t;
                                                                        ## default behavior may use driver-based residency.
        ("pKernelName", c_char_p)                                       ## [in] null-terminated name of kernel in module
    ]

###############################################################################
## @brief Kernel indirect access flags
class ze_kernel_indirect_access_flags_v(IntEnum):
    HOST = ZE_BIT(0)                                                        ## Indicates that the kernel accesses host allocations indirectly.
    DEVICE = ZE_BIT(1)                                                      ## Indicates that the kernel accesses device allocations indirectly.
    SHARED = ZE_BIT(2)                                                      ## Indicates that the kernel accesses shared allocations indirectly.

class ze_kernel_indirect_access_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Supported Cache Config flags
class ze_cache_config_flags_v(IntEnum):
    LARGE_SLM = ZE_BIT(0)                                                   ## Large SLM size
    LARGE_DATA = ZE_BIT(1)                                                  ## Large General Data size

class ze_cache_config_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Maximum kernel universal unique id (UUID) size in bytes
ZE_MAX_KERNEL_UUID_SIZE = 16

###############################################################################
## @brief Maximum module universal unique id (UUID) size in bytes
ZE_MAX_MODULE_UUID_SIZE = 16

###############################################################################
## @brief Kernel universal unique id (UUID)
class ze_kernel_uuid_t(Structure):
    _fields_ = [
        ("kid", c_ubyte * ZE_MAX_KERNEL_UUID_SIZE),                     ## [out] opaque data representing a kernel UUID
        ("mid", c_ubyte * ZE_MAX_MODULE_UUID_SIZE)                      ## [out] opaque data representing the kernel's module UUID
    ]

###############################################################################
## @brief Kernel properties
class ze_kernel_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("numKernelArgs", c_ulong),                                     ## [out] number of kernel arguments.
        ("requiredGroupSizeX", c_ulong),                                ## [out] required group size in the X dimension,
                                                                        ## or zero if there is no required group size
        ("requiredGroupSizeY", c_ulong),                                ## [out] required group size in the Y dimension,
                                                                        ## or zero if there is no required group size
        ("requiredGroupSizeZ", c_ulong),                                ## [out] required group size in the Z dimension,
                                                                        ## or zero if there is no required group size
        ("requiredNumSubGroups", c_ulong),                              ## [out] required number of subgroups per thread group,
                                                                        ## or zero if there is no required number of subgroups
        ("requiredSubgroupSize", c_ulong),                              ## [out] required subgroup size,
                                                                        ## or zero if there is no required subgroup size
        ("maxSubgroupSize", c_ulong),                                   ## [out] maximum subgroup size
        ("maxNumSubgroups", c_ulong),                                   ## [out] maximum number of subgroups per thread group
        ("localMemSize", c_ulong),                                      ## [out] local memory size used by each thread group
        ("privateMemSize", c_ulong),                                    ## [out] private memory size allocated by compiler used by each thread
        ("spillMemSize", c_ulong),                                      ## [out] spill memory size allocated by compiler
        ("uuid", ze_kernel_uuid_t)                                      ## [out] universal unique identifier.
    ]

###############################################################################
## @brief Additional kernel preferred group size properties
## 
## @details
##     - This structure may be passed to ::zeKernelGetProperties, via the
##       `pNext` member of ::ze_kernel_properties_t, to query additional kernel
##       preferred group size properties.
class ze_kernel_preferred_group_size_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("preferredMultiple", c_ulong)                                  ## [out] preferred group size multiple
    ]

###############################################################################
## @brief Kernel dispatch group count.
class ze_group_count_t(Structure):
    _fields_ = [
        ("groupCountX", c_ulong),                                       ## [in] number of thread groups in X dimension
        ("groupCountY", c_ulong),                                       ## [in] number of thread groups in Y dimension
        ("groupCountZ", c_ulong)                                        ## [in] number of thread groups in Z dimension
    ]

###############################################################################
## @brief Module Program Extension Name
ZE_MODULE_PROGRAM_EXP_NAME = "ZE_experimental_module_program"

###############################################################################
## @brief Module Program Extension Version(s)
class ze_module_program_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_module_program_exp_version_t(c_int):
    def __str__(self):
        return str(ze_module_program_exp_version_v(self.value))


###############################################################################
## @brief Module extended descriptor to support multiple input modules.
## 
## @details
##     - Implementation must support ::ZE_experimental_module_program extension
##     - Modules support import and export linkage for functions and global
##       variables.
##     - SPIR-V import and export linkage types are used. See SPIR-V
##       specification for linkage details.
##     - pInputModules, pBuildFlags, and pConstants from ::ze_module_desc_t is
##       ignored.
##     - Format in ::ze_module_desc_t needs to be set to
##       ::ZE_MODULE_FORMAT_IL_SPIRV.
class ze_module_program_exp_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("count", c_ulong),                                             ## [in] Count of input modules
        ("inputSizes", POINTER(c_size_t)),                              ## [in][range(0, count)] sizes of each input IL module in pInputModules.
        ("pInputModules", POINTER(c_ubyte*)),                           ## [in][range(0, count)] pointer to an array of IL (e.g. SPIR-V modules).
                                                                        ## Valid only for SPIR-V input.
        ("pBuildFlags", POINTER(c_char_p)),                             ## [in][optional][range(0, count)] array of strings containing build
                                                                        ## flags. See pBuildFlags in ::ze_module_desc_t.
        ("pConstants", POINTER(ze_module_constants_t*))                 ## [in][optional][range(0, count)] pointer to array of specialization
                                                                        ## constant strings. Valid only for SPIR-V input. This must be set to
                                                                        ## nullptr if no specialization constants are provided.
    ]

###############################################################################
## @brief Raytracing Extension Name
ZE_RAYTRACING_EXT_NAME = "ZE_extension_raytracing"

###############################################################################
## @brief Raytracing Extension Version(s)
class ze_raytracing_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_raytracing_ext_version_t(c_int):
    def __str__(self):
        return str(ze_raytracing_ext_version_v(self.value))


###############################################################################
## @brief Supported raytracing capability flags
class ze_device_raytracing_ext_flags_v(IntEnum):
    RAYQUERY = ZE_BIT(0)                                                    ## Supports rayquery

class ze_device_raytracing_ext_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Raytracing properties queried using ::zeDeviceGetModuleProperties
## 
## @details
##     - This structure may be returned from ::zeDeviceGetModuleProperties, via
##       the `pNext` member of ::ze_device_module_properties_t.
class ze_device_raytracing_ext_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_device_raytracing_ext_flags_t),                    ## [out] 0 or a valid combination of ::ze_device_raytracing_ext_flags_t
        ("maxBVHLevels", c_ulong)                                       ## [out] Maximum number of BVH levels supported
    ]

###############################################################################
## @brief Supported raytracing memory allocation flags
class ze_raytracing_mem_alloc_ext_flags_v(IntEnum):
    TBD = ZE_BIT(0)                                                         ## reserved for future use

class ze_raytracing_mem_alloc_ext_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Raytracing memory allocation descriptor
## 
## @details
##     - This structure must be passed to ::zeMemAllocShared or
##       ::zeMemAllocDevice, via the `pNext` member of
##       ::ze_device_mem_alloc_desc_t, for any memory allocation that is to be
##       accessed by raytracing fixed-function of the device.
class ze_raytracing_mem_alloc_ext_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_raytracing_mem_alloc_ext_flags_t)                  ## [in] flags specifying additional allocation controls.
                                                                        ## must be 0 (default) or a valid combination of ::ze_raytracing_mem_alloc_ext_flag_t;
                                                                        ## default behavior may use implicit driver-based heuristics.
    ]

###############################################################################
## @brief Sampler addressing modes
class ze_sampler_address_mode_v(IntEnum):
    NONE = 0                                                                ## No coordinate modifications for out-of-bounds image access.
    REPEAT = 1                                                              ## Out-of-bounds coordinates are wrapped back around.
    CLAMP = 2                                                               ## Out-of-bounds coordinates are clamped to edge.
    CLAMP_TO_BORDER = 3                                                     ## Out-of-bounds coordinates are clamped to border color which is (0.0f,
                                                                            ## 0.0f, 0.0f, 0.0f) if image format swizzle contains alpha, otherwise
                                                                            ## (0.0f, 0.0f, 0.0f, 1.0f).
    MIRROR = 4                                                              ## Out-of-bounds coordinates are mirrored starting from edge.

class ze_sampler_address_mode_t(c_int):
    def __str__(self):
        return str(ze_sampler_address_mode_v(self.value))


###############################################################################
## @brief Sampler filtering modes
class ze_sampler_filter_mode_v(IntEnum):
    NEAREST = 0                                                             ## No coordinate modifications for out of bounds image access.
    LINEAR = 1                                                              ## Out-of-bounds coordinates are wrapped back around.

class ze_sampler_filter_mode_t(c_int):
    def __str__(self):
        return str(ze_sampler_filter_mode_v(self.value))


###############################################################################
## @brief Sampler descriptor
class ze_sampler_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("addressMode", ze_sampler_address_mode_t),                     ## [in] Sampler addressing mode to determine how out-of-bounds
                                                                        ## coordinates are handled.
        ("filterMode", ze_sampler_filter_mode_t),                       ## [in] Sampler filter mode to determine how samples are filtered.
        ("isNormalized", ze_bool_t)                                     ## [in] Are coordinates normalized [0, 1] or not.
    ]

###############################################################################
## @brief Virtual memory page access attributes
class ze_memory_access_attribute_v(IntEnum):
    NONE = 0                                                                ## Indicates the memory page is inaccessible.
    READWRITE = 1                                                           ## Indicates the memory page supports read write access.
    READONLY = 2                                                            ## Indicates the memory page supports read-only access.

class ze_memory_access_attribute_t(c_int):
    def __str__(self):
        return str(ze_memory_access_attribute_v(self.value))


###############################################################################
## @brief Supported physical memory creation flags
class ze_physical_mem_flags_v(IntEnum):
    ALLOCATE_ON_DEVICE = ZE_BIT(0)                                          ## [default] allocate physical device memory.
    ALLOCATE_ON_HOST = ZE_BIT(1)                                            ## Allocate physical host memory instead.

class ze_physical_mem_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Physical memory descriptor
class ze_physical_mem_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_physical_mem_flags_t),                             ## [in] creation flags.
                                                                        ## must be 0 (default) or a valid combination of
                                                                        ## ::ze_physical_mem_flag_t; default is to create physical device memory.
        ("size", c_size_t)                                              ## [in] size in bytes to reserve; must be page aligned.
    ]

###############################################################################
## @brief Floating-Point Atomics Extension Name
ZE_FLOAT_ATOMICS_EXT_NAME = "ZE_extension_float_atomics"

###############################################################################
## @brief Floating-Point Atomics Extension Version(s)
class ze_float_atomics_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_float_atomics_ext_version_t(c_int):
    def __str__(self):
        return str(ze_float_atomics_ext_version_v(self.value))


###############################################################################
## @brief Supported floating-point atomic capability flags
class ze_device_fp_atomic_ext_flags_v(IntEnum):
    GLOBAL_LOAD_STORE = ZE_BIT(0)                                           ## Supports atomic load, store, and exchange
    GLOBAL_ADD = ZE_BIT(1)                                                  ## Supports atomic add and subtract
    GLOBAL_MIN_MAX = ZE_BIT(2)                                              ## Supports atomic min and max
    LOCAL_LOAD_STORE = ZE_BIT(16)                                           ## Supports atomic load, store, and exchange
    LOCAL_ADD = ZE_BIT(17)                                                  ## Supports atomic add and subtract
    LOCAL_MIN_MAX = ZE_BIT(18)                                              ## Supports atomic min and max

class ze_device_fp_atomic_ext_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Device floating-point atomic properties queried using
##        ::zeDeviceGetModuleProperties
## 
## @details
##     - This structure may be returned from ::zeDeviceGetModuleProperties, via
##       the `pNext` member of ::ze_device_module_properties_t.
class ze_float_atomic_ext_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("fp16Flags", ze_device_fp_atomic_ext_flags_t),                 ## [out] Capabilities for half-precision floating-point atomic operations
        ("fp32Flags", ze_device_fp_atomic_ext_flags_t),                 ## [out] Capabilities for single-precision floating-point atomic
                                                                        ## operations
        ("fp64Flags", ze_device_fp_atomic_ext_flags_t)                  ## [out] Capabilities for double-precision floating-point atomic
                                                                        ## operations
    ]

###############################################################################
## @brief Global Offset Extension Name
ZE_GLOBAL_OFFSET_EXP_NAME = "ZE_experimental_global_offset"

###############################################################################
## @brief Global Offset Extension Version(s)
class ze_global_offset_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_global_offset_exp_version_t(c_int):
    def __str__(self):
        return str(ze_global_offset_exp_version_v(self.value))


###############################################################################
## @brief Relaxed Allocation Limits Extension Name
ZE_RELAXED_ALLOCATION_LIMITS_EXP_NAME = "ZE_experimental_relaxed_allocation_limits"

###############################################################################
## @brief Relaxed Allocation Limits Extension Version(s)
class ze_relaxed_allocation_limits_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_relaxed_allocation_limits_exp_version_t(c_int):
    def __str__(self):
        return str(ze_relaxed_allocation_limits_exp_version_v(self.value))


###############################################################################
## @brief Supported relaxed memory allocation flags
class ze_relaxed_allocation_limits_exp_flags_v(IntEnum):
    MAX_SIZE = ZE_BIT(0)                                                    ## Allocation size may exceed the `maxMemAllocSize` member of
                                                                            ## ::ze_device_properties_t.

class ze_relaxed_allocation_limits_exp_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Relaxed limits memory allocation descriptor
## 
## @details
##     - This structure may be passed to ::zeMemAllocShared or
##       ::zeMemAllocDevice, via the `pNext` member of
##       ::ze_device_mem_alloc_desc_t.
##     - This structure may also be passed to ::zeMemAllocHost, via the `pNext`
##       member of ::ze_host_mem_alloc_desc_t.
class ze_relaxed_allocation_limits_exp_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_relaxed_allocation_limits_exp_flags_t)             ## [in] flags specifying allocation limits to relax.
                                                                        ## must be 0 (default) or a valid combination of ::ze_relaxed_allocation_limits_exp_flag_t;
    ]

###############################################################################
## @brief Get Kernel Binary Extension Name
ZE_GET_KERNEL_BINARY_EXP_NAME = "ZE_extension_kernel_binary_exp"

###############################################################################
## @brief Cache_Reservation Extension Name
ZE_CACHE_RESERVATION_EXT_NAME = "ZE_extension_cache_reservation"

###############################################################################
## @brief Cache_Reservation Extension Version(s)
class ze_cache_reservation_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_cache_reservation_ext_version_t(c_int):
    def __str__(self):
        return str(ze_cache_reservation_ext_version_v(self.value))


###############################################################################
## @brief Cache Reservation Region
class ze_cache_ext_region_v(IntEnum):
    ZE_CACHE_REGION_DEFAULT = 0                                             ## [DEPRECATED] utilize driver default scheme. Use
                                                                            ## ::ZE_CACHE_EXT_REGION_DEFAULT.
    ZE_CACHE_RESERVE_REGION = 1                                             ## [DEPRECATED] utilize reserved region. Use
                                                                            ## ::ZE_CACHE_EXT_REGION_RESERVED.
    ZE_CACHE_NON_RESERVED_REGION = 2                                        ## [DEPRECATED] utilize non-reserverd region. Use
                                                                            ## ::ZE_CACHE_EXT_REGION_NON_RESERVED.
    DEFAULT = 0                                                             ## utilize driver default scheme
    RESERVED = 1                                                            ## utilize reserved region
    NON_RESERVED = 2                                                        ## utilize non-reserverd region

class ze_cache_ext_region_t(c_int):
    def __str__(self):
        return str(ze_cache_ext_region_v(self.value))


###############################################################################
## @brief CacheReservation structure
## 
## @details
##     - This structure must be passed to ::zeDeviceGetCacheProperties via the
##       `pNext` member of ::ze_device_cache_properties_t
##     - Used for determining the max cache reservation allowed on device. Size
##       of zero means no reservation available.
class ze_cache_reservation_ext_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("maxCacheReservationSize", c_size_t)                           ## [out] max cache reservation size
    ]

###############################################################################
## @brief Event Query Timestamps Extension Name
ZE_EVENT_QUERY_TIMESTAMPS_EXP_NAME = "ZE_experimental_event_query_timestamps"

###############################################################################
## @brief Event Query Timestamps Extension Version(s)
class ze_event_query_timestamps_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_event_query_timestamps_exp_version_t(c_int):
    def __str__(self):
        return str(ze_event_query_timestamps_exp_version_v(self.value))


###############################################################################
## @brief Image Memory Properties Extension Name
ZE_IMAGE_MEMORY_PROPERTIES_EXP_NAME = "ZE_experimental_image_memory_properties"

###############################################################################
## @brief Image Memory Properties Extension Version(s)
class ze_image_memory_properties_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_image_memory_properties_exp_version_t(c_int):
    def __str__(self):
        return str(ze_image_memory_properties_exp_version_v(self.value))


###############################################################################
## @brief Image memory properties
class ze_image_memory_properties_exp_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("size", c_ulonglong),                                          ## [out] size of image allocation in bytes.
        ("rowPitch", c_ulonglong),                                      ## [out] size of image row in bytes.
        ("slicePitch", c_ulonglong)                                     ## [out] size of image slice in bytes.
    ]

###############################################################################
## @brief Image View Extension Name
ZE_IMAGE_VIEW_EXT_NAME = "ZE_extension_image_view"

###############################################################################
## @brief Image View Extension Version(s)
class ze_image_view_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_image_view_ext_version_t(c_int):
    def __str__(self):
        return str(ze_image_view_ext_version_v(self.value))


###############################################################################
## @brief Image View Extension Name
ZE_IMAGE_VIEW_EXP_NAME = "ZE_experimental_image_view"

###############################################################################
## @brief Image View Extension Version(s)
class ze_image_view_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_image_view_exp_version_t(c_int):
    def __str__(self):
        return str(ze_image_view_exp_version_v(self.value))


###############################################################################
## @brief Image View Planar Extension Name
ZE_IMAGE_VIEW_PLANAR_EXT_NAME = "ZE_extension_image_view_planar"

###############################################################################
## @brief Image View Planar Extension Version(s)
class ze_image_view_planar_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_image_view_planar_ext_version_t(c_int):
    def __str__(self):
        return str(ze_image_view_planar_ext_version_v(self.value))


###############################################################################
## @brief Image view planar descriptor
class ze_image_view_planar_ext_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("planeIndex", c_ulong)                                         ## [in] the 0-based plane index (e.g. NV12 is 0 = Y plane, 1 UV plane)
    ]

###############################################################################
## @brief Image View Planar Extension Name
ZE_IMAGE_VIEW_PLANAR_EXP_NAME = "ZE_experimental_image_view_planar"

###############################################################################
## @brief Image View Planar Extension Version(s)
class ze_image_view_planar_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_image_view_planar_exp_version_t(c_int):
    def __str__(self):
        return str(ze_image_view_planar_exp_version_v(self.value))


###############################################################################
## @brief Image view planar descriptor
class ze_image_view_planar_exp_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("planeIndex", c_ulong)                                         ## [in] the 0-based plane index (e.g. NV12 is 0 = Y plane, 1 UV plane)
    ]

###############################################################################
## @brief Kernel Scheduling Hints Extension Name
ZE_KERNEL_SCHEDULING_HINTS_EXP_NAME = "ZE_experimental_scheduling_hints"

###############################################################################
## @brief Kernel Scheduling Hints Extension Version(s)
class ze_scheduling_hints_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_scheduling_hints_exp_version_t(c_int):
    def __str__(self):
        return str(ze_scheduling_hints_exp_version_v(self.value))


###############################################################################
## @brief Supported kernel scheduling hint flags
class ze_scheduling_hint_exp_flags_v(IntEnum):
    OLDEST_FIRST = ZE_BIT(0)                                                ## Hint that the kernel prefers oldest-first scheduling
    ROUND_ROBIN = ZE_BIT(1)                                                 ## Hint that the kernel prefers round-robin scheduling
    STALL_BASED_ROUND_ROBIN = ZE_BIT(2)                                     ## Hint that the kernel prefers stall-based round-robin scheduling

class ze_scheduling_hint_exp_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Device kernel scheduling hint properties queried using
##        ::zeDeviceGetModuleProperties
## 
## @details
##     - This structure may be returned from ::zeDeviceGetModuleProperties, via
##       the `pNext` member of ::ze_device_module_properties_t.
class ze_scheduling_hint_exp_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("schedulingHintFlags", ze_scheduling_hint_exp_flags_t)         ## [out] Supported kernel scheduling hints.
                                                                        ## May be 0 (none) or a valid combination of ::ze_scheduling_hint_exp_flag_t.
    ]

###############################################################################
## @brief Kernel scheduling hint descriptor
## 
## @details
##     - This structure may be passed to ::zeKernelSchedulingHintExp.
class ze_scheduling_hint_exp_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_scheduling_hint_exp_flags_t)                       ## [in] flags specifying kernel scheduling hints.
                                                                        ## must be 0 (default) or a valid combination of ::ze_scheduling_hint_exp_flag_t.
    ]

###############################################################################
## @brief Linkonce ODR Extension Name
ZE_LINKONCE_ODR_EXT_NAME = "ZE_extension_linkonce_odr"

###############################################################################
## @brief Linkonce ODR Extension Version(s)
class ze_linkonce_odr_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_linkonce_odr_ext_version_t(c_int):
    def __str__(self):
        return str(ze_linkonce_odr_ext_version_v(self.value))


###############################################################################
## @brief Power Saving Hint Extension Name
ZE_CONTEXT_POWER_SAVING_HINT_EXP_NAME = "ZE_experimental_power_saving_hint"

###############################################################################
## @brief Power Saving Hint Extension Version(s)
class ze_power_saving_hint_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_power_saving_hint_exp_version_t(c_int):
    def __str__(self):
        return str(ze_power_saving_hint_exp_version_v(self.value))


###############################################################################
## @brief Supported device types
class ze_power_saving_hint_type_v(IntEnum):
    MIN = 0                                                                 ## Minumum power savings. The device will make no attempt to save power
                                                                            ## while executing work submitted to this context.
    MAX = 100                                                               ## Maximum power savings. The device will do everything to bring power to
                                                                            ## a minimum while executing work submitted to this context.

class ze_power_saving_hint_type_t(c_int):
    def __str__(self):
        return str(ze_power_saving_hint_type_v(self.value))


###############################################################################
## @brief Extended context descriptor containing power saving hint.
class ze_context_power_saving_hint_exp_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("hint", c_ulong)                                               ## [in] power saving hint (default value = 0). This is value from [0,100]
                                                                        ## and can use pre-defined settings from ::ze_power_saving_hint_type_t.
    ]

###############################################################################
## @brief Subgroups Extension Name
ZE_SUBGROUPS_EXT_NAME = "ZE_extension_subgroups"

###############################################################################
## @brief Subgroups Extension Version(s)
class ze_subgroup_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_subgroup_ext_version_t(c_int):
    def __str__(self):
        return str(ze_subgroup_ext_version_v(self.value))


###############################################################################
## @brief EU Count Extension Name
ZE_EU_COUNT_EXT_NAME = "ZE_extension_eu_count"

###############################################################################
## @brief EU Count Extension Version(s)
class ze_eu_count_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_eu_count_ext_version_t(c_int):
    def __str__(self):
        return str(ze_eu_count_ext_version_v(self.value))


###############################################################################
## @brief EU count queried using ::zeDeviceGetProperties
## 
## @details
##     - This structure may be returned from ::zeDeviceGetProperties via the
##       `pNext` member of ::ze_device_properties_t.
##     - Used for determining the total number of EUs available on device.
class ze_eu_count_ext_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("numTotalEUs", c_ulong)                                        ## [out] Total number of EUs available
    ]

###############################################################################
## @brief PCI Properties Extension Name
ZE_PCI_PROPERTIES_EXT_NAME = "ZE_extension_pci_properties"

###############################################################################
## @brief PCI Properties Extension Version(s)
class ze_pci_properties_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_pci_properties_ext_version_t(c_int):
    def __str__(self):
        return str(ze_pci_properties_ext_version_v(self.value))


###############################################################################
## @brief Device PCI address
## 
## @details
##     - This structure may be passed to ::zeDevicePciGetPropertiesExt as an
##       attribute of ::ze_pci_ext_properties_t.
##     - A PCI BDF address is the bus:device:function address of the device and
##       is useful for locating the device in the PCI switch fabric.
class ze_pci_address_ext_t(Structure):
    _fields_ = [
        ("domain", c_ulong),                                            ## [out] PCI domain number
        ("bus", c_ulong),                                               ## [out] PCI BDF bus number
        ("device", c_ulong),                                            ## [out] PCI BDF device number
        ("function", c_ulong)                                           ## [out] PCI BDF function number
    ]

###############################################################################
## @brief Device PCI speed
class ze_pci_speed_ext_t(Structure):
    _fields_ = [
        ("genVersion", c_int32_t),                                      ## [out] The link generation. A value of -1 means that this property is
                                                                        ## unknown.
        ("width", c_int32_t),                                           ## [out] The number of lanes. A value of -1 means that this property is
                                                                        ## unknown.
        ("maxBandwidth", c_int64_t)                                     ## [out] The theoretical maximum bandwidth in bytes/sec (sum of all
                                                                        ## lanes). A value of -1 means that this property is unknown.
    ]

###############################################################################
## @brief Static PCI properties
class ze_pci_ext_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("address", ze_pci_address_ext_t),                              ## [out] The BDF address
        ("maxSpeed", ze_pci_speed_ext_t)                                ## [out] Fastest port configuration supported by the device (sum of all
                                                                        ## lanes)
    ]

###############################################################################
## @brief sRGB Extension Name
ZE_SRGB_EXT_NAME = "ZE_extension_srgb"

###############################################################################
## @brief sRGB Extension Version(s)
class ze_srgb_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_srgb_ext_version_t(c_int):
    def __str__(self):
        return str(ze_srgb_ext_version_v(self.value))


###############################################################################
## @brief sRGB image descriptor
## 
## @details
##     - This structure may be passed to ::zeImageCreate via the `pNext` member
##       of ::ze_image_desc_t
##     - Used for specifying that the image is in sRGB format.
class ze_srgb_ext_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("sRGB", ze_bool_t)                                             ## [in] Is sRGB.
    ]

###############################################################################
## @brief Image Copy Extension Name
ZE_IMAGE_COPY_EXT_NAME = "ZE_extension_image_copy"

###############################################################################
## @brief Image Copy Extension Version(s)
class ze_image_copy_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_image_copy_ext_version_t(c_int):
    def __str__(self):
        return str(ze_image_copy_ext_version_v(self.value))


###############################################################################
## @brief Image Query Allocation Properties Extension Name
ZE_IMAGE_QUERY_ALLOC_PROPERTIES_EXT_NAME = "ZE_extension_image_query_alloc_properties"

###############################################################################
## @brief Image Query Allocation Properties Extension Version(s)
class ze_image_query_alloc_properties_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_image_query_alloc_properties_ext_version_t(c_int):
    def __str__(self):
        return str(ze_image_query_alloc_properties_ext_version_v(self.value))


###############################################################################
## @brief Image allocation properties queried using
##        ::zeImageGetAllocPropertiesExt
class ze_image_allocation_ext_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("id", c_ulonglong)                                             ## [out] identifier for this allocation
    ]

###############################################################################
## @brief Linkage Inspection Extension Name
ZE_LINKAGE_INSPECTION_EXT_NAME = "ZE_extension_linkage_inspection"

###############################################################################
## @brief Linkage Inspection Extension Version(s)
class ze_linkage_inspection_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_linkage_inspection_ext_version_t(c_int):
    def __str__(self):
        return str(ze_linkage_inspection_ext_version_v(self.value))


###############################################################################
## @brief Supported module linkage inspection flags
class ze_linkage_inspection_ext_flags_v(IntEnum):
    IMPORTS = ZE_BIT(0)                                                     ## List all imports of modules
    UNRESOLVABLE_IMPORTS = ZE_BIT(1)                                        ## List all imports of modules that do not have a corresponding export
    EXPORTS = ZE_BIT(2)                                                     ## List all exports of modules

class ze_linkage_inspection_ext_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Module linkage inspection descriptor
## 
## @details
##     - This structure may be passed to ::zeModuleInspectLinkageExt.
class ze_linkage_inspection_ext_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_linkage_inspection_ext_flags_t)                    ## [in] flags specifying module linkage inspection.
                                                                        ## must be 0 (default) or a valid combination of ::ze_linkage_inspection_ext_flag_t.
    ]

###############################################################################
## @brief Memory Compression Hints Extension Name
ZE_MEMORY_COMPRESSION_HINTS_EXT_NAME = "ZE_extension_memory_compression_hints"

###############################################################################
## @brief Memory Compression Hints Extension Version(s)
class ze_memory_compression_hints_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_memory_compression_hints_ext_version_t(c_int):
    def __str__(self):
        return str(ze_memory_compression_hints_ext_version_v(self.value))


###############################################################################
## @brief Supported memory compression hints flags
class ze_memory_compression_hints_ext_flags_v(IntEnum):
    COMPRESSED = ZE_BIT(0)                                                  ## Hint Driver implementation to make allocation compressible
    UNCOMPRESSED = ZE_BIT(1)                                                ## Hint Driver implementation to make allocation not compressible

class ze_memory_compression_hints_ext_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Compression hints memory allocation descriptor
## 
## @details
##     - This structure may be passed to ::zeMemAllocShared or
##       ::zeMemAllocDevice, via the `pNext` member of
##       ::ze_device_mem_alloc_desc_t.
##     - This structure may be passed to ::zeMemAllocHost, via the `pNext`
##       member of ::ze_host_mem_alloc_desc_t.
##     - This structure may be passed to ::zeImageCreate, via the `pNext`
##       member of ::ze_image_desc_t.
class ze_memory_compression_hints_ext_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_memory_compression_hints_ext_flags_t)              ## [in] flags specifying if allocation should be compressible or not.
                                                                        ## Must be set to one of the ::ze_memory_compression_hints_ext_flag_t;
    ]

###############################################################################
## @brief Memory Free Policies Extension Name
ZE_MEMORY_FREE_POLICIES_EXT_NAME = "ZE_extension_memory_free_policies"

###############################################################################
## @brief Memory Free Policies Extension Version(s)
class ze_memory_free_policies_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_memory_free_policies_ext_version_t(c_int):
    def __str__(self):
        return str(ze_memory_free_policies_ext_version_v(self.value))


###############################################################################
## @brief Supported memory free policy capability flags
class ze_driver_memory_free_policy_ext_flags_v(IntEnum):
    BLOCKING_FREE = ZE_BIT(0)                                               ## blocks until all commands using the memory are complete before freeing
    DEFER_FREE = ZE_BIT(1)                                                  ## schedules the memory to be freed but does not free immediately

class ze_driver_memory_free_policy_ext_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Driver memory free properties queried using ::zeDriverGetProperties
## 
## @details
##     - All drivers must support an immediate free policy, which is the
##       default free policy.
##     - This structure may be returned from ::zeDriverGetProperties, via the
##       `pNext` member of ::ze_driver_properties_t.
class ze_driver_memory_free_ext_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("freePolicies", ze_driver_memory_free_policy_ext_flags_t)      ## [out] Supported memory free policies.
                                                                        ## must be 0 or a combination of ::ze_driver_memory_free_policy_ext_flag_t.
    ]

###############################################################################
## @brief Memory free descriptor with free policy
class ze_memory_free_ext_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("freePolicy", ze_driver_memory_free_policy_ext_flags_t)        ## [in] flags specifying the memory free policy.
                                                                        ## must be 0 (default) or a supported ::ze_driver_memory_free_policy_ext_flag_t;
                                                                        ## default behavior is to free immediately.
    ]

###############################################################################
## @brief Bandwidth Extension Name
ZE_BANDWIDTH_PROPERTIES_EXP_NAME = "ZE_experimental_bandwidth_properties"

###############################################################################
## @brief P2P Bandwidth Properties
## 
## @details
##     - This structure may be passed to ::zeDeviceGetP2PProperties by having
##       the pNext member of ::ze_device_p2p_properties_t point at this struct.
class ze_device_p2p_bandwidth_exp_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("logicalBandwidth", c_ulong),                                  ## [out] total logical design bandwidth for all links connecting the two
                                                                        ## devices
        ("physicalBandwidth", c_ulong),                                 ## [out] total physical design bandwidth for all links connecting the two
                                                                        ## devices
        ("bandwidthUnit", ze_bandwidth_unit_t),                         ## [out] bandwidth unit
        ("logicalLatency", c_ulong),                                    ## [out] average logical design latency for all links connecting the two
                                                                        ## devices
        ("physicalLatency", c_ulong),                                   ## [out] average physical design latency for all links connecting the two
                                                                        ## devices
        ("latencyUnit", ze_latency_unit_t)                              ## [out] latency unit
    ]

###############################################################################
## @brief Copy Bandwidth Properties
## 
## @details
##     - This structure may be passed to
##       ::zeDeviceGetCommandQueueGroupProperties by having the pNext member of
##       ::ze_command_queue_group_properties_t point at this struct.
class ze_copy_bandwidth_exp_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("copyBandwidth", c_ulong),                                     ## [out] design bandwidth supported by this engine type for copy
                                                                        ## operations
        ("copyBandwidthUnit", ze_bandwidth_unit_t)                      ## [out] copy bandwidth unit
    ]

###############################################################################
## @brief Device Local Identifier (LUID) Extension Name
ZE_DEVICE_LUID_EXT_NAME = "ZE_extension_device_luid"

###############################################################################
## @brief Device Local Identifier (LUID) Extension Version(s)
class ze_device_luid_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_device_luid_ext_version_t(c_int):
    def __str__(self):
        return str(ze_device_luid_ext_version_v(self.value))


###############################################################################
## @brief Maximum device local identifier (LUID) size in bytes
ZE_MAX_DEVICE_LUID_SIZE_EXT = 8

###############################################################################
## @brief Device local identifier (LUID)
class ze_device_luid_ext_t(Structure):
    _fields_ = [
        ("id", c_ubyte * ZE_MAX_DEVICE_LUID_SIZE_EXT)                   ## [out] opaque data representing a device LUID
    ]

###############################################################################
## @brief Device LUID properties queried using ::zeDeviceGetProperties
## 
## @details
##     - This structure may be returned from ::zeDeviceGetProperties, via the
##       `pNext` member of ::ze_device_properties_t.
class ze_device_luid_ext_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("luid", ze_device_luid_ext_t),                                 ## [out] locally unique identifier (LUID).
                                                                        ## The returned LUID can be cast to a LUID object and must be equal to
                                                                        ## the locally
                                                                        ## unique identifier of an IDXGIAdapter1 object that corresponds to the device.
        ("nodeMask", c_ulong)                                           ## [out] node mask.
                                                                        ## The returned node mask must contain exactly one bit.
                                                                        ## If the device is running on an operating system that supports the
                                                                        ## Direct3D 12 API
                                                                        ## and the device corresponds to an individual device in a linked device
                                                                        ## adapter, the
                                                                        ## returned node mask identifies the Direct3D 12 node corresponding to
                                                                        ## the device.
                                                                        ## Otherwise, the returned node mask must be 1.
    ]

###############################################################################
## @brief Fabric Topology Discovery Extension Name
ZE_FABRIC_EXP_NAME = "ZE_experimental_fabric"

###############################################################################
## @brief Maximum fabric edge model string size
ZE_MAX_FABRIC_EDGE_MODEL_EXP_SIZE = 256

###############################################################################
## @brief Fabric Vertex types
class ze_fabric_vertex_exp_type_v(IntEnum):
    UNKNOWN = 0                                                             ## Fabric vertex type is unknown
    DEVICE = 1                                                              ## Fabric vertex represents a device
    SUBDEVICE = 2                                                           ## Fabric vertex represents a subdevice
    SWITCH = 3                                                              ## Fabric vertex represents a switch

class ze_fabric_vertex_exp_type_t(c_int):
    def __str__(self):
        return str(ze_fabric_vertex_exp_type_v(self.value))


###############################################################################
## @brief Fabric edge duplexity
class ze_fabric_edge_exp_duplexity_v(IntEnum):
    UNKNOWN = 0                                                             ## Fabric edge duplexity is unknown
    HALF_DUPLEX = 1                                                         ## Fabric edge is half duplex, i.e. stated bandwidth is obtained in only
                                                                            ## one direction at time
    FULL_DUPLEX = 2                                                         ## Fabric edge is full duplex, i.e. stated bandwidth is supported in both
                                                                            ## directions simultaneously

class ze_fabric_edge_exp_duplexity_t(c_int):
    def __str__(self):
        return str(ze_fabric_edge_exp_duplexity_v(self.value))


###############################################################################
## @brief PCI address
## 
## @details
##     - A PCI BDF address is the bus:device:function address of the device and
##       is useful for locating the device in the PCI switch fabric.
class ze_fabric_vertex_pci_exp_address_t(Structure):
    _fields_ = [
        ("domain", c_ulong),                                            ## [out] PCI domain number
        ("bus", c_ulong),                                               ## [out] PCI BDF bus number
        ("device", c_ulong),                                            ## [out] PCI BDF device number
        ("function", c_ulong)                                           ## [out] PCI BDF function number
    ]

###############################################################################
## @brief Fabric Vertex properties
class ze_fabric_vertex_exp_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("uuid", ze_uuid_t),                                            ## [out] universal unique identifier. If the vertex is co-located with a
                                                                        ## device/subdevice, then this uuid will match that of the corresponding
                                                                        ## device/subdevice
        ("type", ze_fabric_vertex_exp_type_t),                          ## [out] does the fabric vertex represent a device, subdevice, or switch?
        ("remote", ze_bool_t),                                          ## [out] does the fabric vertex live on the local node or on a remote
                                                                        ## node?
        ("address", ze_fabric_vertex_pci_exp_address_t)                 ## [out] B/D/F address of fabric vertex & associated device/subdevice if
                                                                        ## available
    ]

###############################################################################
## @brief Fabric Edge properties
class ze_fabric_edge_exp_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("uuid", ze_uuid_t),                                            ## [out] universal unique identifier.
        ("model", c_char * ZE_MAX_FABRIC_EDGE_MODEL_EXP_SIZE),          ## [out] Description of fabric edge technology. Will be set to the string
                                                                        ## "unkown" if this cannot be determined for this edge
        ("bandwidth", c_ulong),                                         ## [out] design bandwidth
        ("bandwidthUnit", ze_bandwidth_unit_t),                         ## [out] bandwidth unit
        ("latency", c_ulong),                                           ## [out] design latency
        ("latencyUnit", ze_latency_unit_t),                             ## [out] latency unit
        ("duplexity", ze_fabric_edge_exp_duplexity_t)                   ## [out] Duplexity of the fabric edge
    ]

###############################################################################
## @brief Device Memory Properties Extension Name
ZE_DEVICE_MEMORY_PROPERTIES_EXT_NAME = "ZE_extension_device_memory_properties"

###############################################################################
## @brief Device Memory Properties Extension Version(s)
class ze_device_memory_properties_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_device_memory_properties_ext_version_t(c_int):
    def __str__(self):
        return str(ze_device_memory_properties_ext_version_v(self.value))


###############################################################################
## @brief Memory module types
class ze_device_memory_ext_type_v(IntEnum):
    HBM = 0                                                                 ## HBM memory
    HBM2 = 1                                                                ## HBM2 memory
    DDR = 2                                                                 ## DDR memory
    DDR2 = 3                                                                ## DDR2 memory
    DDR3 = 4                                                                ## DDR3 memory
    DDR4 = 5                                                                ## DDR4 memory
    DDR5 = 6                                                                ## DDR5 memory
    LPDDR = 7                                                               ## LPDDR memory
    LPDDR3 = 8                                                              ## LPDDR3 memory
    LPDDR4 = 9                                                              ## LPDDR4 memory
    LPDDR5 = 10                                                             ## LPDDR5 memory
    SRAM = 11                                                               ## SRAM memory
    L1 = 12                                                                 ## L1 cache
    L3 = 13                                                                 ## L3 cache
    GRF = 14                                                                ## Execution unit register file
    SLM = 15                                                                ## Execution unit shared local memory
    GDDR4 = 16                                                              ## GDDR4 memory
    GDDR5 = 17                                                              ## GDDR5 memory
    GDDR5X = 18                                                             ## GDDR5X memory
    GDDR6 = 19                                                              ## GDDR6 memory
    GDDR6X = 20                                                             ## GDDR6X memory
    GDDR7 = 21                                                              ## GDDR7 memory

class ze_device_memory_ext_type_t(c_int):
    def __str__(self):
        return str(ze_device_memory_ext_type_v(self.value))


###############################################################################
## @brief Memory properties
## 
## @details
##     - This structure may be returned from ::zeDeviceGetMemoryProperties via
##       the `pNext` member of ::ze_device_memory_properties_t
class ze_device_memory_ext_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("type", ze_device_memory_ext_type_t),                          ## [out] The memory type
        ("physicalSize", c_ulonglong),                                  ## [out] Physical memory size in bytes. A value of 0 indicates that this
                                                                        ## property is not known. However, a call to ::zesMemoryGetState() will
                                                                        ## correctly return the total size of usable memory.
        ("readBandwidth", c_ulong),                                     ## [out] Design bandwidth for reads
        ("writeBandwidth", c_ulong),                                    ## [out] Design bandwidth for writes
        ("bandwidthUnit", ze_bandwidth_unit_t)                          ## [out] bandwidth unit
    ]

###############################################################################
## @brief Bfloat16 Conversions Extension Name
ZE_BFLOAT16_CONVERSIONS_EXT_NAME = "ZE_extension_bfloat16_conversions"

###############################################################################
## @brief Bfloat16 Conversions Extension Version(s)
class ze_bfloat16_conversions_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_bfloat16_conversions_ext_version_t(c_int):
    def __str__(self):
        return str(ze_bfloat16_conversions_ext_version_v(self.value))


###############################################################################
## @brief Device IP Version Extension Name
ZE_DEVICE_IP_VERSION_EXT_NAME = "ZE_extension_device_ip_version"

###############################################################################
## @brief Device IP Version Extension Version(s)
class ze_device_ip_version_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_device_ip_version_version_t(c_int):
    def __str__(self):
        return str(ze_device_ip_version_version_v(self.value))


###############################################################################
## @brief Device IP version queried using ::zeDeviceGetProperties
## 
## @details
##     - This structure may be returned from ::zeDeviceGetProperties via the
##       `pNext` member of ::ze_device_properties_t
class ze_device_ip_version_ext_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("ipVersion", c_ulong)                                          ## [out] Device IP version. The meaning of the device IP version is
                                                                        ## implementation-defined, but newer devices should have a higher
                                                                        ## version than older devices.
    ]

###############################################################################
## @brief Kernel Max Group Size Properties Extension Name
ZE_KERNEL_MAX_GROUP_SIZE_PROPERTIES_EXT_NAME = "ZE_extension_kernel_max_group_size_properties"

###############################################################################
## @brief Kernel Max Group Size Properties Extension Version(s)
class ze_kernel_max_group_size_properties_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_kernel_max_group_size_properties_ext_version_t(c_int):
    def __str__(self):
        return str(ze_kernel_max_group_size_properties_ext_version_v(self.value))


###############################################################################
## @brief Additional kernel max group size properties
## 
## @details
##     - This structure may be passed to ::zeKernelGetProperties, via the
##       `pNext` member of ::ze_kernel_properties_t, to query additional kernel
##       max group size properties.
class ze_kernel_max_group_size_properties_ext_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("maxGroupSize", c_ulong)                                       ## [out] maximum group size that can be used to execute the kernel. This
                                                                        ## value may be less than or equal to the `maxTotalGroupSize` member of
                                                                        ## ::ze_device_compute_properties_t.
    ]

###############################################################################
## @brief compiler-independent type
class ze_kernel_max_group_size_ext_properties_t(ze_kernel_max_group_size_properties_ext_t):
    pass

###############################################################################
## @brief Sub-Allocations Properties Extension Name
ZE_SUB_ALLOCATIONS_EXP_NAME = "ZE_experimental_sub_allocations"

###############################################################################
## @brief Sub-Allocations Properties Extension Version(s)
class ze_sub_allocations_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_sub_allocations_exp_version_t(c_int):
    def __str__(self):
        return str(ze_sub_allocations_exp_version_v(self.value))


###############################################################################
## @brief Properties returned for a sub-allocation
class ze_sub_allocation_t(Structure):
    _fields_ = [
        ("base", c_void_p),                                             ## [in,out][optional] base address of the sub-allocation
        ("size", c_size_t)                                              ## [in,out][optional] size of the allocation
    ]

###############################################################################
## @brief Sub-Allocations Properties
## 
## @details
##     - This structure may be passed to ::zeMemGetAllocProperties, via the
##       `pNext` member of ::ze_memory_allocation_properties_t.
class ze_memory_sub_allocations_exp_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("pCount", POINTER(c_ulong)),                                   ## [in,out] pointer to the number of sub-allocations.
                                                                        ## if count is zero, then the driver shall update the value with the
                                                                        ## total number of sub-allocations on which the allocation has been divided.
                                                                        ## if count is greater than the number of sub-allocations, then the
                                                                        ## driver shall update the value with the correct number of sub-allocations.
        ("pSubAllocations", POINTER(ze_sub_allocation_t))               ## [in,out][optional][range(0, *pCount)] array of properties for sub-allocations.
                                                                        ## if count is less than the number of sub-allocations available, then
                                                                        ## driver shall only retrieve properties for that number of sub-allocations.
    ]

###############################################################################
## @brief Event Query Kernel Timestamps Extension Name
ZE_EVENT_QUERY_KERNEL_TIMESTAMPS_EXT_NAME = "ZE_extension_event_query_kernel_timestamps"

###############################################################################
## @brief Event Query Kernel Timestamps Extension Version(s)
class ze_event_query_kernel_timestamps_ext_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_event_query_kernel_timestamps_ext_version_t(c_int):
    def __str__(self):
        return str(ze_event_query_kernel_timestamps_ext_version_v(self.value))


###############################################################################
## @brief Event query kernel timestamps flags
class ze_event_query_kernel_timestamps_ext_flags_v(IntEnum):
    KERNEL = ZE_BIT(0)                                                      ## Kernel timestamp results
    SYNCHRONIZED = ZE_BIT(1)                                                ## Device event timestamps synchronized to the host time domain

class ze_event_query_kernel_timestamps_ext_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Event query kernel timestamps properties
## 
## @details
##     - This structure may be returned from ::zeDeviceGetProperties, via the
##       `pNext` member of ::ze_device_properties_t.
class ze_event_query_kernel_timestamps_ext_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_event_query_kernel_timestamps_ext_flags_t)         ## [out] 0 or some combination of
                                                                        ## ::ze_event_query_kernel_timestamps_ext_flag_t flags
    ]

###############################################################################
## @brief Kernel timestamp clock data synchronized to the host time domain
class ze_synchronized_timestamp_data_ext_t(Structure):
    _fields_ = [
        ("kernelStart", c_ulonglong),                                   ## [out] synchronized clock at start of kernel execution
        ("kernelEnd", c_ulonglong)                                      ## [out] synchronized clock at end of kernel execution
    ]

###############################################################################
## @brief Synchronized kernel timestamp result
class ze_synchronized_timestamp_result_ext_t(Structure):
    _fields_ = [
        ("global", ze_synchronized_timestamp_data_ext_t),               ## [out] wall-clock data
        ("context", ze_synchronized_timestamp_data_ext_t)               ## [out] context-active data; only includes clocks while device context
                                                                        ## was actively executing.
    ]

###############################################################################
## @brief Event query kernel timestamps results properties
class ze_event_query_kernel_timestamps_results_ext_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("pKernelTimestampsBuffer", POINTER(ze_kernel_timestamp_result_t)), ## [in,out][optional][range(0, *pCount)] pointer to destination buffer of
                                                                        ## kernel timestamp results
        ("pSynchronizedTimestampsBuffer", POINTER(ze_synchronized_timestamp_result_ext_t))  ## [in,out][optional][range(0, *pCount)] pointer to destination buffer of
                                                                        ## synchronized timestamp results
    ]

###############################################################################
## @brief Ray Tracing Acceleration Structure Builder Extension Name
ZE_RTAS_BUILDER_EXP_NAME = "ZE_experimental_rtas_builder"

###############################################################################
## @brief Ray Tracing Acceleration Structure Builder Extension Version(s)
class ze_rtas_builder_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_rtas_builder_exp_version_t(c_int):
    def __str__(self):
        return str(ze_rtas_builder_exp_version_v(self.value))


###############################################################################
## @brief Ray tracing acceleration structure device flags
class ze_rtas_device_exp_flags_v(IntEnum):
    RESERVED = ZE_BIT(0)                                                    ## reserved for future use

class ze_rtas_device_exp_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Ray tracing acceleration structure format
## 
## @details
##     - This is an opaque ray tracing acceleration structure format
##       identifier.
class ze_rtas_format_exp_v(IntEnum):
    INVALID = 0                                                             ## Invalid acceleration structure format

class ze_rtas_format_exp_t(c_int):
    def __str__(self):
        return str(ze_rtas_format_exp_v(self.value))


###############################################################################
## @brief Ray tracing acceleration structure builder flags
class ze_rtas_builder_exp_flags_v(IntEnum):
    RESERVED = ZE_BIT(0)                                                    ## Reserved for future use

class ze_rtas_builder_exp_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Ray tracing acceleration structure builder parallel operation flags
class ze_rtas_parallel_operation_exp_flags_v(IntEnum):
    RESERVED = ZE_BIT(0)                                                    ## Reserved for future use

class ze_rtas_parallel_operation_exp_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Ray tracing acceleration structure builder geometry flags
class ze_rtas_builder_geometry_exp_flags_v(IntEnum):
    NON_OPAQUE = ZE_BIT(0)                                                  ## non-opaque geometries invoke an any-hit shader

class ze_rtas_builder_geometry_exp_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Packed ray tracing acceleration structure builder geometry flags (see
##        ::ze_rtas_builder_geometry_exp_flags_t)
class ze_rtas_builder_packed_geometry_exp_flags_t(c_ubyte):
    pass

###############################################################################
## @brief Ray tracing acceleration structure builder instance flags
class ze_rtas_builder_instance_exp_flags_v(IntEnum):
    TRIANGLE_CULL_DISABLE = ZE_BIT(0)                                       ## disables culling of front-facing and back-facing triangles
    TRIANGLE_FRONT_COUNTERCLOCKWISE = ZE_BIT(1)                             ## reverses front and back face of triangles
    TRIANGLE_FORCE_OPAQUE = ZE_BIT(2)                                       ## forces instanced geometry to be opaque, unless ray flag forces it to
                                                                            ## be non-opaque
    TRIANGLE_FORCE_NON_OPAQUE = ZE_BIT(3)                                   ## forces instanced geometry to be non-opaque, unless ray flag forces it
                                                                            ## to be opaque

class ze_rtas_builder_instance_exp_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Packed ray tracing acceleration structure builder instance flags (see
##        ::ze_rtas_builder_instance_exp_flags_t)
class ze_rtas_builder_packed_instance_exp_flags_t(c_ubyte):
    pass

###############################################################################
## @brief Ray tracing acceleration structure builder build operation flags
## 
## @details
##     - These flags allow the application to tune the acceleration structure
##       build operation.
##     - The acceleration structure builder implementation might choose to use
##       spatial splitting to split large or long primitives into smaller
##       pieces. This may result in any-hit shaders being invoked multiple
##       times for non-opaque primitives, unless
##       ::ZE_RTAS_BUILDER_BUILD_OP_EXP_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION is specified.
##     - Usage of any of these flags may reduce ray tracing performance.
class ze_rtas_builder_build_op_exp_flags_v(IntEnum):
    COMPACT = ZE_BIT(0)                                                     ## build more compact acceleration structure
    NO_DUPLICATE_ANYHIT_INVOCATION = ZE_BIT(1)                              ## guarantees single any-hit shader invocation per primitive

class ze_rtas_builder_build_op_exp_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Ray tracing acceleration structure builder build quality hint
## 
## @details
##     - Depending on use case different quality modes for acceleration
##       structure build are supported.
##     - A low-quality build builds an acceleration structure fast, but at the
##       cost of some reduction in ray tracing performance. This mode is
##       recommended for dynamic content, such as animated characters.
##     - A medium-quality build uses a compromise between build quality and ray
##       tracing performance. This mode should be used by default.
##     - Higher ray tracing performance can be achieved by using a high-quality
##       build, but acceleration structure build performance might be
##       significantly reduced.
class ze_rtas_builder_build_quality_hint_exp_v(IntEnum):
    LOW = 0                                                                 ## build low-quality acceleration structure (fast)
    MEDIUM = 1                                                              ## build medium-quality acceleration structure (slower)
    HIGH = 2                                                                ## build high-quality acceleration structure (slow)

class ze_rtas_builder_build_quality_hint_exp_t(c_int):
    def __str__(self):
        return str(ze_rtas_builder_build_quality_hint_exp_v(self.value))


###############################################################################
## @brief Ray tracing acceleration structure builder geometry type
class ze_rtas_builder_geometry_type_exp_v(IntEnum):
    TRIANGLES = 0                                                           ## triangle mesh geometry type
    QUADS = 1                                                               ## quad mesh geometry type
    PROCEDURAL = 2                                                          ## procedural geometry type
    INSTANCE = 3                                                            ## instance geometry type

class ze_rtas_builder_geometry_type_exp_t(c_int):
    def __str__(self):
        return str(ze_rtas_builder_geometry_type_exp_v(self.value))


###############################################################################
## @brief Packed ray tracing acceleration structure builder geometry type (see
##        ::ze_rtas_builder_geometry_type_exp_t)
class ze_rtas_builder_packed_geometry_type_exp_t(c_ubyte):
    pass

###############################################################################
## @brief Ray tracing acceleration structure data buffer element format
## 
## @details
##     - Specifies the format of data buffer elements.
##     - Data buffers may contain instancing transform matrices, triangle/quad
##       vertex indices, etc...
class ze_rtas_builder_input_data_format_exp_v(IntEnum):
    FLOAT3 = 0                                                              ## 3-component float vector (see ::ze_rtas_float3_exp_t)
    FLOAT3X4_COLUMN_MAJOR = 1                                               ## 3x4 affine transformation in column-major format (see
                                                                            ## ::ze_rtas_transform_float3x4_column_major_exp_t)
    FLOAT3X4_ALIGNED_COLUMN_MAJOR = 2                                       ## 3x4 affine transformation in column-major format (see
                                                                            ## ::ze_rtas_transform_float3x4_aligned_column_major_exp_t)
    FLOAT3X4_ROW_MAJOR = 3                                                  ## 3x4 affine transformation in row-major format (see
                                                                            ## ::ze_rtas_transform_float3x4_row_major_exp_t)
    AABB = 4                                                                ## 3-dimensional axis-aligned bounding-box (see ::ze_rtas_aabb_exp_t)
    TRIANGLE_INDICES_UINT32 = 5                                             ## Unsigned 32-bit triangle indices (see
                                                                            ## ::ze_rtas_triangle_indices_uint32_exp_t)
    QUAD_INDICES_UINT32 = 6                                                 ## Unsigned 32-bit quad indices (see ::ze_rtas_quad_indices_uint32_exp_t)

class ze_rtas_builder_input_data_format_exp_t(c_int):
    def __str__(self):
        return str(ze_rtas_builder_input_data_format_exp_v(self.value))


###############################################################################
## @brief Packed ray tracing acceleration structure data buffer element format
##        (see ::ze_rtas_builder_input_data_format_exp_t)
class ze_rtas_builder_packed_input_data_format_exp_t(c_ubyte):
    pass

###############################################################################
## @brief Handle of ray tracing acceleration structure builder object
class ze_rtas_builder_exp_handle_t(c_void_p):
    pass

###############################################################################
## @brief Handle of ray tracing acceleration structure builder parallel
##        operation object
class ze_rtas_parallel_operation_exp_handle_t(c_void_p):
    pass

###############################################################################
## @brief Ray tracing acceleration structure builder descriptor
class ze_rtas_builder_exp_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("builderVersion", ze_rtas_builder_exp_version_t)               ## [in] ray tracing acceleration structure builder version
    ]

###############################################################################
## @brief Ray tracing acceleration structure builder properties
class ze_rtas_builder_exp_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_rtas_builder_exp_flags_t),                         ## [out] ray tracing acceleration structure builder flags
        ("rtasBufferSizeBytesExpected", c_size_t),                      ## [out] expected size (in bytes) required for acceleration structure buffer
                                                                        ##    - When using an acceleration structure buffer of this size, the
                                                                        ## build is expected to succeed; however, it is possible that the build
                                                                        ## may fail with ::ZE_RESULT_EXP_RTAS_BUILD_RETRY
        ("rtasBufferSizeBytesMaxRequired", c_size_t),                   ## [out] worst-case size (in bytes) required for acceleration structure buffer
                                                                        ##    - When using an acceleration structure buffer of this size, the
                                                                        ## build is guaranteed to not run out of memory.
        ("scratchBufferSizeBytes", c_size_t)                            ## [out] scratch buffer size (in bytes) required for acceleration
                                                                        ## structure build.
    ]

###############################################################################
## @brief Ray tracing acceleration structure builder parallel operation
##        properties
class ze_rtas_parallel_operation_exp_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_rtas_parallel_operation_exp_flags_t),              ## [out] ray tracing acceleration structure builder parallel operation
                                                                        ## flags
        ("maxConcurrency", c_ulong)                                     ## [out] maximum number of threads that may join the parallel operation
    ]

###############################################################################
## @brief Ray tracing acceleration structure device properties
## 
## @details
##     - This structure may be passed to ::zeDeviceGetProperties, via `pNext`
##       member of ::ze_device_properties_t.
##     - The implementation shall populate `format` with a value other than
##       ::ZE_RTAS_FORMAT_EXP_INVALID when the device supports ray tracing.
class ze_rtas_device_exp_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_rtas_device_exp_flags_t),                          ## [out] ray tracing acceleration structure device flags
        ("rtasFormat", ze_rtas_format_exp_t),                           ## [out] ray tracing acceleration structure format
        ("rtasBufferAlignment", c_ulong)                                ## [out] required alignment of acceleration structure buffer
    ]

###############################################################################
## @brief A 3-component vector type
class ze_rtas_float3_exp_t(Structure):
    _fields_ = [
        ("x", c_float),                                                 ## [in] x-coordinate of float3 vector
        ("y", c_float),                                                 ## [in] y-coordinate of float3 vector
        ("z", c_float)                                                  ## [in] z-coordinate of float3 vector
    ]

###############################################################################
## @brief 3x4 affine transformation in column-major layout
## 
## @details
##     - A 3x4 affine transformation in column major layout, consisting of vectors
##          - vx=(vx_x, vx_y, vx_z),
##          - vy=(vy_x, vy_y, vy_z),
##          - vz=(vz_x, vz_y, vz_z), and
##          - p=(p_x, p_y, p_z)
##     - The transformation transforms a point (x, y, z) to: `x*vx + y*vy +
##       z*vz + p`.
class ze_rtas_transform_float3x4_column_major_exp_t(Structure):
    _fields_ = [
        ("vx_x", c_float),                                              ## [in] element 0 of column 0 of 3x4 matrix
        ("vx_y", c_float),                                              ## [in] element 1 of column 0 of 3x4 matrix
        ("vx_z", c_float),                                              ## [in] element 2 of column 0 of 3x4 matrix
        ("vy_x", c_float),                                              ## [in] element 0 of column 1 of 3x4 matrix
        ("vy_y", c_float),                                              ## [in] element 1 of column 1 of 3x4 matrix
        ("vy_z", c_float),                                              ## [in] element 2 of column 1 of 3x4 matrix
        ("vz_x", c_float),                                              ## [in] element 0 of column 2 of 3x4 matrix
        ("vz_y", c_float),                                              ## [in] element 1 of column 2 of 3x4 matrix
        ("vz_z", c_float),                                              ## [in] element 2 of column 2 of 3x4 matrix
        ("p_x", c_float),                                               ## [in] element 0 of column 3 of 3x4 matrix
        ("p_y", c_float),                                               ## [in] element 1 of column 3 of 3x4 matrix
        ("p_z", c_float)                                                ## [in] element 2 of column 3 of 3x4 matrix
    ]

###############################################################################
## @brief 3x4 affine transformation in column-major layout with aligned column
##        vectors
## 
## @details
##     - A 3x4 affine transformation in column major layout, consisting of vectors
##        - vx=(vx_x, vx_y, vx_z),
##        - vy=(vy_x, vy_y, vy_z),
##        - vz=(vz_x, vz_y, vz_z), and
##        - p=(p_x, p_y, p_z)
##     - The transformation transforms a point (x, y, z) to: `x*vx + y*vy +
##       z*vz + p`.
##     - The column vectors are aligned to 16-bytes and pad members are
##       ignored.
class ze_rtas_transform_float3x4_aligned_column_major_exp_t(Structure):
    _fields_ = [
        ("vx_x", c_float),                                              ## [in] element 0 of column 0 of 3x4 matrix
        ("vx_y", c_float),                                              ## [in] element 1 of column 0 of 3x4 matrix
        ("vx_z", c_float),                                              ## [in] element 2 of column 0 of 3x4 matrix
        ("pad0", c_float),                                              ## [in] ignored padding
        ("vy_x", c_float),                                              ## [in] element 0 of column 1 of 3x4 matrix
        ("vy_y", c_float),                                              ## [in] element 1 of column 1 of 3x4 matrix
        ("vy_z", c_float),                                              ## [in] element 2 of column 1 of 3x4 matrix
        ("pad1", c_float),                                              ## [in] ignored padding
        ("vz_x", c_float),                                              ## [in] element 0 of column 2 of 3x4 matrix
        ("vz_y", c_float),                                              ## [in] element 1 of column 2 of 3x4 matrix
        ("vz_z", c_float),                                              ## [in] element 2 of column 2 of 3x4 matrix
        ("pad2", c_float),                                              ## [in] ignored padding
        ("p_x", c_float),                                               ## [in] element 0 of column 3 of 3x4 matrix
        ("p_y", c_float),                                               ## [in] element 1 of column 3 of 3x4 matrix
        ("p_z", c_float),                                               ## [in] element 2 of column 3 of 3x4 matrix
        ("pad3", c_float)                                               ## [in] ignored padding
    ]

###############################################################################
## @brief 3x4 affine transformation in row-major layout
## 
## @details
##     - A 3x4 affine transformation in row-major layout, consisting of vectors
##          - vx=(vx_x, vx_y, vx_z),
##          - vy=(vy_x, vy_y, vy_z),
##          - vz=(vz_x, vz_y, vz_z), and
##          - p=(p_x, p_y, p_z)
##     - The transformation transforms a point (x, y, z) to: `x*vx + y*vy +
##       z*vz + p`.
class ze_rtas_transform_float3x4_row_major_exp_t(Structure):
    _fields_ = [
        ("vx_x", c_float),                                              ## [in] element 0 of row 0 of 3x4 matrix
        ("vy_x", c_float),                                              ## [in] element 1 of row 0 of 3x4 matrix
        ("vz_x", c_float),                                              ## [in] element 2 of row 0 of 3x4 matrix
        ("p_x", c_float),                                               ## [in] element 3 of row 0 of 3x4 matrix
        ("vx_y", c_float),                                              ## [in] element 0 of row 1 of 3x4 matrix
        ("vy_y", c_float),                                              ## [in] element 1 of row 1 of 3x4 matrix
        ("vz_y", c_float),                                              ## [in] element 2 of row 1 of 3x4 matrix
        ("p_y", c_float),                                               ## [in] element 3 of row 1 of 3x4 matrix
        ("vx_z", c_float),                                              ## [in] element 0 of row 2 of 3x4 matrix
        ("vy_z", c_float),                                              ## [in] element 1 of row 2 of 3x4 matrix
        ("vz_z", c_float),                                              ## [in] element 2 of row 2 of 3x4 matrix
        ("p_z", c_float)                                                ## [in] element 3 of row 2 of 3x4 matrix
    ]

###############################################################################
## @brief A 3-dimensional axis-aligned bounding-box with lower and upper bounds
##        in each dimension
class ze_rtas_aabb_exp_t(Structure):
    _fields_ = [
        ("lower", ze_rtas_c_float3_exp_t),                              ## [in] lower bounds of AABB
        ("upper", ze_rtas_c_float3_exp_t)                               ## [in] upper bounds of AABB
    ]

###############################################################################
## @brief Triangle represented using 3 vertex indices
## 
## @details
##     - Represents a triangle using 3 vertex indices that index into a vertex
##       array that needs to be provided together with the index array.
##     - The linear barycentric u/v parametrization of the triangle is defined as:
##          - (u=0, v=0) at v0,
##          - (u=1, v=0) at v1, and
##          - (u=0, v=1) at v2
class ze_rtas_triangle_indices_uint32_exp_t(Structure):
    _fields_ = [
        ("v0", c_ulong),                                                ## [in] first index pointing to the first triangle vertex in vertex array
        ("v1", c_ulong),                                                ## [in] second index pointing to the second triangle vertex in vertex
                                                                        ## array
        ("v2", c_ulong)                                                 ## [in] third index pointing to the third triangle vertex in vertex array
    ]

###############################################################################
## @brief Quad represented using 4 vertex indices
## 
## @details
##     - Represents a quad composed of 4 indices that index into a vertex array
##       that needs to be provided together with the index array.
##     - A quad is a triangle pair represented using 4 vertex indices v0, v1,
##       v2, v3.
##       The first triangle is made out of indices v0, v1, v3 and the second triangle
##       from indices v2, v3, v1. The piecewise linear barycentric u/v parametrization
##       of the quad is defined as:
##          - (u=0, v=0) at v0,
##          - (u=1, v=0) at v1,
##          - (u=0, v=1) at v3, and
##          - (u=1, v=1) at v2
##       This is achieved by correcting the u'/v' coordinates of the second
##       triangle by
##       *u = 1-u'* and *v = 1-v'*, yielding a piecewise linear parametrization.
class ze_rtas_quad_indices_uint32_exp_t(Structure):
    _fields_ = [
        ("v0", c_ulong),                                                ## [in] first index pointing to the first quad vertex in vertex array
        ("v1", c_ulong),                                                ## [in] second index pointing to the second quad vertex in vertex array
        ("v2", c_ulong),                                                ## [in] third index pointing to the third quad vertex in vertex array
        ("v3", c_ulong)                                                 ## [in] fourth index pointing to the fourth quad vertex in vertex array
    ]

###############################################################################
## @brief Ray tracing acceleration structure builder geometry info
class ze_rtas_builder_geometry_info_exp_t(Structure):
    _fields_ = [
        ("geometryType", ze_rtas_builder_packed_geometry_type_exp_t)    ## [in] geometry type
    ]

###############################################################################
## @brief Ray tracing acceleration structure builder triangle mesh geometry info
## 
## @details
##     - The linear barycentric u/v parametrization of the triangle is defined as:
##          - (u=0, v=0) at v0,
##          - (u=1, v=0) at v1, and
##          - (u=0, v=1) at v2
class ze_rtas_builder_triangles_geometry_info_exp_t(Structure):
    _fields_ = [
        ("geometryType", ze_rtas_builder_packed_geometry_type_exp_t),   ## [in] geometry type, must be
                                                                        ## ::ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_TRIANGLES
        ("geometryFlags", ze_rtas_builder_packed_geometry_exp_flags_t), ## [in] 0 or some combination of ::ze_rtas_builder_geometry_exp_flag_t
                                                                        ## bits representing the geometry flags for all primitives of this
                                                                        ## geometry
        ("geometryMask", c_ubyte),                                      ## [in] 8-bit geometry mask for ray masking
        ("triangleFormat", ze_rtas_builder_packed_input_data_format_exp_t), ## [in] format of triangle buffer data, must be
                                                                        ## ::ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_TRIANGLE_INDICES_UINT32
        ("vertexFormat", ze_rtas_builder_packed_input_data_format_exp_t),   ## [in] format of vertex buffer data, must be
                                                                        ## ::ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3
        ("triangleCount", c_ulong),                                     ## [in] number of triangles in triangle buffer
        ("vertexCount", c_ulong),                                       ## [in] number of vertices in vertex buffer
        ("triangleStride", c_ulong),                                    ## [in] stride (in bytes) of triangles in triangle buffer
        ("vertexStride", c_ulong),                                      ## [in] stride (in bytes) of vertices in vertex buffer
        ("pTriangleBuffer", c_void_p),                                  ## [in] pointer to array of triangle indices in specified format
        ("pVertexBuffer", c_void_p)                                     ## [in] pointer to array of triangle vertices in specified format
    ]

###############################################################################
## @brief Ray tracing acceleration structure builder quad mesh geometry info
## 
## @details
##     - A quad is a triangle pair represented using 4 vertex indices v0, v1,
##       v2, v3.
##       The first triangle is made out of indices v0, v1, v3 and the second triangle
##       from indices v2, v3, v1. The piecewise linear barycentric u/v parametrization
##       of the quad is defined as:
##          - (u=0, v=0) at v0,
##          - (u=1, v=0) at v1,
##          - (u=0, v=1) at v3, and
##          - (u=1, v=1) at v2
##       This is achieved by correcting the u'/v' coordinates of the second
##       triangle by
##       *u = 1-u'* and *v = 1-v'*, yielding a piecewise linear parametrization.
class ze_rtas_builder_quads_geometry_info_exp_t(Structure):
    _fields_ = [
        ("geometryType", ze_rtas_builder_packed_geometry_type_exp_t),   ## [in] geometry type, must be ::ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_QUADS
        ("geometryFlags", ze_rtas_builder_packed_geometry_exp_flags_t), ## [in] 0 or some combination of ::ze_rtas_builder_geometry_exp_flag_t
                                                                        ## bits representing the geometry flags for all primitives of this
                                                                        ## geometry
        ("geometryMask", c_ubyte),                                      ## [in] 8-bit geometry mask for ray masking
        ("quadFormat", ze_rtas_builder_packed_input_data_format_exp_t), ## [in] format of quad buffer data, must be
                                                                        ## ::ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_QUAD_INDICES_UINT32
        ("vertexFormat", ze_rtas_builder_packed_input_data_format_exp_t),   ## [in] format of vertex buffer data, must be
                                                                        ## ::ZE_RTAS_BUILDER_INPUT_DATA_FORMAT_EXP_FLOAT3
        ("quadCount", c_ulong),                                         ## [in] number of quads in quad buffer
        ("vertexCount", c_ulong),                                       ## [in] number of vertices in vertex buffer
        ("quadStride", c_ulong),                                        ## [in] stride (in bytes) of quads in quad buffer
        ("vertexStride", c_ulong),                                      ## [in] stride (in bytes) of vertices in vertex buffer
        ("pQuadBuffer", c_void_p),                                      ## [in] pointer to array of quad indices in specified format
        ("pVertexBuffer", c_void_p)                                     ## [in] pointer to array of quad vertices in specified format
    ]

###############################################################################
## @brief AABB callback function parameters
class ze_rtas_geometry_aabbs_exp_cb_params_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("primID", c_ulong),                                            ## [in] first primitive to return bounds for
        ("primIDCount", c_ulong),                                       ## [in] number of primitives to return bounds for
        ("pGeomUserPtr", c_void_p),                                     ## [in] pointer provided through geometry descriptor
        ("pBuildUserPtr", c_void_p),                                    ## [in] pointer provided through ::zeRTASBuilderBuildExp function
        ("pBoundsOut", POINTER(ze_rtas_aabb_exp_t))                     ## [out] destination buffer to write AABB bounds to
    ]

###############################################################################
## @brief Callback function pointer type to return AABBs for a range of
##        procedural primitives

###############################################################################
## @brief Ray tracing acceleration structure builder procedural primitives
##        geometry info
## 
## @details
##     - A host-side bounds callback function is invoked by the acceleration
##       structure builder to query the bounds of procedural primitives on
##       demand. The callback is passed some `pGeomUserPtr` that can point to
##       an application-side representation of the procedural primitives.
##       Further, a second `pBuildUserPtr`, which is set by a parameter to
##       ::zeRTASBuilderBuildExp, is passed to the callback. This allows the
##       build to change the bounds of the procedural geometry, for example, to
##       build a BVH only over a short time range to implement multi-segment
##       motion blur.
class ze_rtas_builder_procedural_geometry_info_exp_t(Structure):
    _fields_ = [
        ("geometryType", ze_rtas_builder_packed_geometry_type_exp_t),   ## [in] geometry type, must be
                                                                        ## ::ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_PROCEDURAL
        ("geometryFlags", ze_rtas_builder_packed_geometry_exp_flags_t), ## [in] 0 or some combination of ::ze_rtas_builder_geometry_exp_flag_t
                                                                        ## bits representing the geometry flags for all primitives of this
                                                                        ## geometry
        ("geometryMask", c_ubyte),                                      ## [in] 8-bit geometry mask for ray masking
        ("reserved", c_ubyte),                                          ## [in] reserved for future use
        ("primCount", c_ulong),                                         ## [in] number of primitives in geometry
        ("pfnGetBoundsCb", ze_rtas_geometry_aabbs_cb_exp_t),            ## [in] pointer to callback function to get the axis-aligned bounding-box
                                                                        ## for a range of primitives
        ("pGeomUserPtr", c_void_p)                                      ## [in] user data pointer passed to callback
    ]

###############################################################################
## @brief Ray tracing acceleration structure builder instance geometry info
class ze_rtas_builder_instance_geometry_info_exp_t(Structure):
    _fields_ = [
        ("geometryType", ze_rtas_builder_packed_geometry_type_exp_t),   ## [in] geometry type, must be
                                                                        ## ::ZE_RTAS_BUILDER_GEOMETRY_TYPE_EXP_INSTANCE
        ("instanceFlags", ze_rtas_builder_packed_instance_exp_flags_t), ## [in] 0 or some combination of ::ze_rtas_builder_geometry_exp_flag_t
                                                                        ## bits representing the geometry flags for all primitives of this
                                                                        ## geometry
        ("geometryMask", c_ubyte),                                      ## [in] 8-bit geometry mask for ray masking
        ("transformFormat", ze_rtas_builder_packed_input_data_format_exp_t),## [in] format of the specified transformation
        ("instanceUserID", c_ulong),                                    ## [in] user-specified identifier for the instance
        ("pTransform", c_void_p),                                       ## [in] object-to-world instance transformation in specified format
        ("pBounds", POINTER(ze_rtas_aabb_exp_t)),                       ## [in] object-space axis-aligned bounding-box of the instanced
                                                                        ## acceleration structure
        ("pAccelerationStructure", c_void_p)                            ## [in] pointer to acceleration structure to instantiate
    ]

###############################################################################
## @brief 
class ze_rtas_builder_build_op_exp_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("rtasFormat", ze_rtas_format_exp_t),                           ## [in] ray tracing acceleration structure format
        ("buildQuality", ze_rtas_builder_build_quality_hint_exp_t),     ## [in] acceleration structure build quality hint
        ("buildFlags", ze_rtas_builder_build_op_exp_flags_t),           ## [in] 0 or some combination of ::ze_rtas_builder_build_op_exp_flag_t
                                                                        ## flags
        ("ppGeometries", POINTER(ze_rtas_builder_geometry_info_exp_t*)),## [in][optional][range(0, `numGeometries`)] NULL or a valid array of
                                                                        ## pointers to geometry infos
        ("numGeometries", c_ulong)                                      ## [in] number of geometries in geometry infos array, can be zero when
                                                                        ## `ppGeometries` is NULL
    ]

###############################################################################
## @brief Counter-based Event Pools Extension Name
ZE_EVENT_POOL_COUNTER_BASED_EXP_NAME = "ZE_experimental_event_pool_counter_based"

###############################################################################
## @brief Counter-based Event Pools Extension Version(s)
class ze_event_pool_counter_based_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_event_pool_counter_based_exp_version_t(c_int):
    def __str__(self):
        return str(ze_event_pool_counter_based_exp_version_v(self.value))


###############################################################################
## @brief Supported event flags for defining counter-based event pools.
class ze_event_pool_counter_based_exp_flags_v(IntEnum):
    IMMEDIATE = ZE_BIT(0)                                                   ## Counter-based event pool is used for immediate command lists (default)
    NON_IMMEDIATE = ZE_BIT(1)                                               ## Counter-based event pool is for non-immediate command lists

class ze_event_pool_counter_based_exp_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Event pool descriptor for counter-based events. This structure may be
##        passed to ::zeEventPoolCreate as pNext member of
##        ::ze_event_pool_desc_t.
class ze_event_pool_counter_based_exp_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_event_pool_counter_based_exp_flags_t)              ## [in] mode flags.
                                                                        ## must be 0 (default) or a valid value of ::ze_event_pool_counter_based_exp_flag_t
                                                                        ## default behavior is counter-based event pool is only used for
                                                                        ## immediate command lists.
    ]

###############################################################################
## @brief Image Memory Properties Extension Name
ZE_BINDLESS_IMAGE_EXP_NAME = "ZE_experimental_bindless_image"

###############################################################################
## @brief Bindless Image Extension Version(s)
class ze_bindless_image_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_bindless_image_exp_version_t(c_int):
    def __str__(self):
        return str(ze_bindless_image_exp_version_v(self.value))


###############################################################################
## @brief Image flags for Bindless images
class ze_image_bindless_exp_flags_v(IntEnum):
    BINDLESS = ZE_BIT(0)                                                    ## Bindless images are created with ::zeImageCreate. The image handle
                                                                            ## created with this flag is valid on both host and device.
    SAMPLED_IMAGE = ZE_BIT(1)                                               ## Bindless sampled images are created with ::zeImageCreate by combining
                                                                            ## BINDLESS and SAMPLED_IMAGE.
                                                                            ## Create sampled image view from bindless unsampled image using SAMPLED_IMAGE.

class ze_image_bindless_exp_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Image descriptor for bindless images. This structure may be passed to
##        ::zeImageCreate via pNext member of ::ze_image_desc_t.
class ze_image_bindless_exp_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_image_bindless_exp_flags_t)                        ## [in] image flags.
                                                                        ## must be 0 (default) or a valid value of ::ze_image_bindless_exp_flag_t
                                                                        ## default behavior is bindless images are not used when creating handles
                                                                        ## via ::zeImageCreate.
                                                                        ## When the flag is passed to ::zeImageCreate, then only the memory for
                                                                        ## the image is allocated.
                                                                        ## Additional image handles can be created with ::zeImageViewCreateExt.
                                                                        ## When ::ZE_IMAGE_BINDLESS_EXP_FLAG_SAMPLED_IMAGE flag is passed,
                                                                        ## ::ze_sampler_desc_t must be attached via pNext member of ::ze_image_bindless_exp_desc_t.
    ]

###############################################################################
## @brief Image descriptor for bindless images created from pitched allocations.
##        This structure may be passed to ::zeImageCreate via pNext member of
##        ::ze_image_desc_t.
class ze_image_pitched_exp_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("ptr", c_void_p)                                               ## [in] pointer to pitched device allocation allocated using ::zeMemAllocDevice
    ]

###############################################################################
## @brief Device specific properties for pitched allocations
## 
## @details
##     - This structure may be passed to ::zeDeviceGetImageProperties via the
##       pNext member of ::ze_device_image_properties_t.
class ze_device_pitched_alloc_exp_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("maxImageLinearWidth", c_size_t),                              ## [out] Maximum image linear width.
        ("maxImageLinearHeight", c_size_t)                              ## [out] Maximum image linear height.
    ]

###############################################################################
## @brief Command List Clone Extension Name
ZE_COMMAND_LIST_CLONE_EXP_NAME = "ZE_experimental_command_list_clone"

###############################################################################
## @brief Command List Clone Extension Version(s)
class ze_command_list_clone_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_command_list_clone_exp_version_t(c_int):
    def __str__(self):
        return str(ze_command_list_clone_exp_version_v(self.value))


###############################################################################
## @brief Immediate Command List Append Extension Name
ZE_IMMEDIATE_COMMAND_LIST_APPEND_EXP_NAME = "ZE_experimental_immediate_command_list_append"

###############################################################################
## @brief Immediate Command List Append Extension Version(s)
class ze_immediate_command_list_append_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    CURRENT = ZE_MAKE_VERSION( 1, 0 )                                       ## latest known version

class ze_immediate_command_list_append_exp_version_t(c_int):
    def __str__(self):
        return str(ze_immediate_command_list_append_exp_version_v(self.value))


###############################################################################
## @brief Mutable Command List Extension Name
ZE_MUTABLE_COMMAND_LIST_EXP_NAME = "ZE_experimental_mutable_command_list"

###############################################################################
## @brief Mutable Command List Extension Version(s)
class ze_mutable_command_list_exp_version_v(IntEnum):
    _1_0 = ZE_MAKE_VERSION( 1, 0 )                                          ## version 1.0
    _1_1 = ZE_MAKE_VERSION( 1, 1 )                                          ## version 1.1
    CURRENT = ZE_MAKE_VERSION( 1, 1 )                                       ## latest known version

class ze_mutable_command_list_exp_version_t(c_int):
    def __str__(self):
        return str(ze_mutable_command_list_exp_version_v(self.value))


###############################################################################
## @brief Mutable command flags
class ze_mutable_command_exp_flags_v(IntEnum):
    KERNEL_ARGUMENTS = ZE_BIT(0)                                            ## kernel arguments
    GROUP_COUNT = ZE_BIT(1)                                                 ## kernel group count
    GROUP_SIZE = ZE_BIT(2)                                                  ## kernel group size
    GLOBAL_OFFSET = ZE_BIT(3)                                               ## kernel global offset
    SIGNAL_EVENT = ZE_BIT(4)                                                ## command signal event
    WAIT_EVENTS = ZE_BIT(5)                                                 ## command wait events
    KERNEL_INSTRUCTION = ZE_BIT(6)                                          ## command kernel
    GRAPH_ARGUMENTS = ZE_BIT(7)                                             ## graph arguments

class ze_mutable_command_exp_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Mutable command identifier descriptor
class ze_mutable_command_id_exp_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_mutable_command_exp_flags_t)                       ## [in] mutable command flags.
                                                                        ##  - must be 0 (default, equivalent to setting all flags bar kernel
                                                                        ## instruction), or a valid combination of ::ze_mutable_command_exp_flag_t
                                                                        ##  - in order to include kernel instruction mutation,
                                                                        ## ::ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_INSTRUCTION must be explictly included
    ]

###############################################################################
## @brief Mutable command list flags
class ze_mutable_command_list_exp_flags_v(IntEnum):
    RESERVED = ZE_BIT(0)                                                    ## reserved

class ze_mutable_command_list_exp_flags_t(c_int):
    def __str__(self):
        return hex(self.value)


###############################################################################
## @brief Mutable command list properties
class ze_mutable_command_list_exp_properties_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in,out][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("mutableCommandListFlags", ze_mutable_command_list_exp_flags_t),   ## [out] mutable command list flags
        ("mutableCommandFlags", ze_mutable_command_exp_flags_t)         ## [out] mutable command flags
    ]

###############################################################################
## @brief Mutable command list descriptor
class ze_mutable_command_list_exp_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", ze_mutable_command_list_exp_flags_t)                  ## [in] mutable command list flags.
                                                                        ##  - must be 0 (default) or a valid combination of ::ze_mutable_command_list_exp_flag_t
    ]

###############################################################################
## @brief Mutable commands descriptor
class ze_mutable_commands_exp_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("flags", c_ulong)                                              ## [in] must be 0, this field is reserved for future use
    ]

###############################################################################
## @brief Mutable kernel argument descriptor
class ze_mutable_kernel_argument_exp_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("commandId", c_ulonglong),                                     ## [in] command identifier
        ("argIndex", c_ulong),                                          ## [in] kernel argument index
        ("argSize", c_size_t),                                          ## [in] kernel argument size
        ("pArgValue", c_void_p)                                         ## [in] pointer to kernel argument value
    ]

###############################################################################
## @brief Mutable kernel group count descriptor
class ze_mutable_group_count_exp_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("commandId", c_ulonglong),                                     ## [in] command identifier
        ("pGroupCount", POINTER(ze_group_count_t))                      ## [in] pointer to group count
    ]

###############################################################################
## @brief Mutable kernel group size descriptor
class ze_mutable_group_size_exp_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("commandId", c_ulonglong),                                     ## [in] command identifier
        ("groupSizeX", c_ulong),                                        ## [in] group size for X dimension to use for the kernel
        ("groupSizeY", c_ulong),                                        ## [in] group size for Y dimension to use for the kernel
        ("groupSizeZ", c_ulong)                                         ## [in] group size for Z dimension to use for the kernel
    ]

###############################################################################
## @brief Mutable kernel global offset descriptor
class ze_mutable_global_offset_exp_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("commandId", c_ulonglong),                                     ## [in] command identifier
        ("offsetX", c_ulong),                                           ## [in] global offset for X dimension to use for this kernel
        ("offsetY", c_ulong),                                           ## [in] global offset for Y dimension to use for this kernel
        ("offsetZ", c_ulong)                                            ## [in] global offset for Z dimension to use for this kernel
    ]

###############################################################################
## @brief Mutable graph argument descriptor
class ze_mutable_graph_argument_exp_desc_t(Structure):
    _fields_ = [
        ("stype", ze_structure_type_t),                                 ## [in] type of this structure
        ("pNext", c_void_p),                                            ## [in][optional] must be null or a pointer to an extension-specific
                                                                        ## structure (i.e. contains stype and pNext).
        ("commandId", c_ulonglong),                                     ## [in] command identifier
        ("argIndex", c_ulong),                                          ## [in] graph argument index
        ("pArgValue", c_void_p)                                         ## [in] pointer to graph argument value
    ]

###############################################################################
__use_win_types = "Windows" == platform.uname()[0]

###############################################################################
## @brief Function-pointer for zeRTASBuilderCreateExp
if __use_win_types:
    _zeRTASBuilderCreateExp_t = WINFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(ze_rtas_builder_exp_desc_t), POINTER(ze_rtas_builder_exp_handle_t) )
else:
    _zeRTASBuilderCreateExp_t = CFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(ze_rtas_builder_exp_desc_t), POINTER(ze_rtas_builder_exp_handle_t) )

###############################################################################
## @brief Function-pointer for zeRTASBuilderGetBuildPropertiesExp
if __use_win_types:
    _zeRTASBuilderGetBuildPropertiesExp_t = WINFUNCTYPE( ze_result_t, ze_rtas_builder_exp_handle_t, POINTER(ze_rtas_builder_build_op_exp_desc_t), POINTER(ze_rtas_builder_exp_properties_t) )
else:
    _zeRTASBuilderGetBuildPropertiesExp_t = CFUNCTYPE( ze_result_t, ze_rtas_builder_exp_handle_t, POINTER(ze_rtas_builder_build_op_exp_desc_t), POINTER(ze_rtas_builder_exp_properties_t) )

###############################################################################
## @brief Function-pointer for zeRTASBuilderBuildExp
if __use_win_types:
    _zeRTASBuilderBuildExp_t = WINFUNCTYPE( ze_result_t, ze_rtas_builder_exp_handle_t, POINTER(ze_rtas_builder_build_op_exp_desc_t), c_void_p, c_size_t, c_void_p, c_size_t, ze_rtas_parallel_operation_exp_handle_t, c_void_p, POINTER(ze_rtas_aabb_exp_t), POINTER(c_size_t) )
else:
    _zeRTASBuilderBuildExp_t = CFUNCTYPE( ze_result_t, ze_rtas_builder_exp_handle_t, POINTER(ze_rtas_builder_build_op_exp_desc_t), c_void_p, c_size_t, c_void_p, c_size_t, ze_rtas_parallel_operation_exp_handle_t, c_void_p, POINTER(ze_rtas_aabb_exp_t), POINTER(c_size_t) )

###############################################################################
## @brief Function-pointer for zeRTASBuilderDestroyExp
if __use_win_types:
    _zeRTASBuilderDestroyExp_t = WINFUNCTYPE( ze_result_t, ze_rtas_builder_exp_handle_t )
else:
    _zeRTASBuilderDestroyExp_t = CFUNCTYPE( ze_result_t, ze_rtas_builder_exp_handle_t )


###############################################################################
## @brief Table of RTASBuilderExp functions pointers
class _ze_rtas_builder_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnCreateExp", c_void_p),                                     ## _zeRTASBuilderCreateExp_t
        ("pfnGetBuildPropertiesExp", c_void_p),                         ## _zeRTASBuilderGetBuildPropertiesExp_t
        ("pfnBuildExp", c_void_p),                                      ## _zeRTASBuilderBuildExp_t
        ("pfnDestroyExp", c_void_p)                                     ## _zeRTASBuilderDestroyExp_t
    ]

###############################################################################
## @brief Function-pointer for zeRTASParallelOperationCreateExp
if __use_win_types:
    _zeRTASParallelOperationCreateExp_t = WINFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(ze_rtas_parallel_operation_exp_handle_t) )
else:
    _zeRTASParallelOperationCreateExp_t = CFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(ze_rtas_parallel_operation_exp_handle_t) )

###############################################################################
## @brief Function-pointer for zeRTASParallelOperationGetPropertiesExp
if __use_win_types:
    _zeRTASParallelOperationGetPropertiesExp_t = WINFUNCTYPE( ze_result_t, ze_rtas_parallel_operation_exp_handle_t, POINTER(ze_rtas_parallel_operation_exp_properties_t) )
else:
    _zeRTASParallelOperationGetPropertiesExp_t = CFUNCTYPE( ze_result_t, ze_rtas_parallel_operation_exp_handle_t, POINTER(ze_rtas_parallel_operation_exp_properties_t) )

###############################################################################
## @brief Function-pointer for zeRTASParallelOperationJoinExp
if __use_win_types:
    _zeRTASParallelOperationJoinExp_t = WINFUNCTYPE( ze_result_t, ze_rtas_parallel_operation_exp_handle_t )
else:
    _zeRTASParallelOperationJoinExp_t = CFUNCTYPE( ze_result_t, ze_rtas_parallel_operation_exp_handle_t )

###############################################################################
## @brief Function-pointer for zeRTASParallelOperationDestroyExp
if __use_win_types:
    _zeRTASParallelOperationDestroyExp_t = WINFUNCTYPE( ze_result_t, ze_rtas_parallel_operation_exp_handle_t )
else:
    _zeRTASParallelOperationDestroyExp_t = CFUNCTYPE( ze_result_t, ze_rtas_parallel_operation_exp_handle_t )


###############################################################################
## @brief Table of RTASParallelOperationExp functions pointers
class _ze_rtas_parallel_operation_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnCreateExp", c_void_p),                                     ## _zeRTASParallelOperationCreateExp_t
        ("pfnGetPropertiesExp", c_void_p),                              ## _zeRTASParallelOperationGetPropertiesExp_t
        ("pfnJoinExp", c_void_p),                                       ## _zeRTASParallelOperationJoinExp_t
        ("pfnDestroyExp", c_void_p)                                     ## _zeRTASParallelOperationDestroyExp_t
    ]

###############################################################################
## @brief Function-pointer for zeInit
if __use_win_types:
    _zeInit_t = WINFUNCTYPE( ze_result_t, ze_init_flags_t )
else:
    _zeInit_t = CFUNCTYPE( ze_result_t, ze_init_flags_t )

###############################################################################
## @brief Function-pointer for zeInitDrivers
if __use_win_types:
    _zeInitDrivers_t = WINFUNCTYPE( ze_result_t, POINTER(c_ulong), POINTER(ze_driver_handle_t), POINTER(ze_init_driver_type_desc_t) )
else:
    _zeInitDrivers_t = CFUNCTYPE( ze_result_t, POINTER(c_ulong), POINTER(ze_driver_handle_t), POINTER(ze_init_driver_type_desc_t) )


###############################################################################
## @brief Table of Global functions pointers
class _ze_global_dditable_t(Structure):
    _fields_ = [
        ("pfnInit", c_void_p),                                          ## _zeInit_t
        ("pfnInitDrivers", c_void_p)                                    ## _zeInitDrivers_t
    ]

###############################################################################
## @brief Function-pointer for zeDriverGet
if __use_win_types:
    _zeDriverGet_t = WINFUNCTYPE( ze_result_t, POINTER(c_ulong), POINTER(ze_driver_handle_t) )
else:
    _zeDriverGet_t = CFUNCTYPE( ze_result_t, POINTER(c_ulong), POINTER(ze_driver_handle_t) )

###############################################################################
## @brief Function-pointer for zeDriverGetApiVersion
if __use_win_types:
    _zeDriverGetApiVersion_t = WINFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(ze_api_version_t) )
else:
    _zeDriverGetApiVersion_t = CFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(ze_api_version_t) )

###############################################################################
## @brief Function-pointer for zeDriverGetProperties
if __use_win_types:
    _zeDriverGetProperties_t = WINFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(ze_driver_properties_t) )
else:
    _zeDriverGetProperties_t = CFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(ze_driver_properties_t) )

###############################################################################
## @brief Function-pointer for zeDriverGetIpcProperties
if __use_win_types:
    _zeDriverGetIpcProperties_t = WINFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(ze_driver_ipc_properties_t) )
else:
    _zeDriverGetIpcProperties_t = CFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(ze_driver_ipc_properties_t) )

###############################################################################
## @brief Function-pointer for zeDriverGetExtensionProperties
if __use_win_types:
    _zeDriverGetExtensionProperties_t = WINFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(c_ulong), POINTER(ze_driver_extension_properties_t) )
else:
    _zeDriverGetExtensionProperties_t = CFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(c_ulong), POINTER(ze_driver_extension_properties_t) )

###############################################################################
## @brief Function-pointer for zeDriverGetExtensionFunctionAddress
if __use_win_types:
    _zeDriverGetExtensionFunctionAddress_t = WINFUNCTYPE( ze_result_t, ze_driver_handle_t, c_char_p, POINTER(c_void_p) )
else:
    _zeDriverGetExtensionFunctionAddress_t = CFUNCTYPE( ze_result_t, ze_driver_handle_t, c_char_p, POINTER(c_void_p) )

###############################################################################
## @brief Function-pointer for zeDriverGetLastErrorDescription
if __use_win_types:
    _zeDriverGetLastErrorDescription_t = WINFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(c_char_p) )
else:
    _zeDriverGetLastErrorDescription_t = CFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(c_char_p) )


###############################################################################
## @brief Table of Driver functions pointers
class _ze_driver_dditable_t(Structure):
    _fields_ = [
        ("pfnGet", c_void_p),                                           ## _zeDriverGet_t
        ("pfnGetApiVersion", c_void_p),                                 ## _zeDriverGetApiVersion_t
        ("pfnGetProperties", c_void_p),                                 ## _zeDriverGetProperties_t
        ("pfnGetIpcProperties", c_void_p),                              ## _zeDriverGetIpcProperties_t
        ("pfnGetExtensionProperties", c_void_p),                        ## _zeDriverGetExtensionProperties_t
        ("pfnGetExtensionFunctionAddress", c_void_p),                   ## _zeDriverGetExtensionFunctionAddress_t
        ("pfnGetLastErrorDescription", c_void_p)                        ## _zeDriverGetLastErrorDescription_t
    ]

###############################################################################
## @brief Function-pointer for zeDriverRTASFormatCompatibilityCheckExp
if __use_win_types:
    _zeDriverRTASFormatCompatibilityCheckExp_t = WINFUNCTYPE( ze_result_t, ze_driver_handle_t, ze_rtas_format_exp_t, ze_rtas_format_exp_t )
else:
    _zeDriverRTASFormatCompatibilityCheckExp_t = CFUNCTYPE( ze_result_t, ze_driver_handle_t, ze_rtas_format_exp_t, ze_rtas_format_exp_t )


###############################################################################
## @brief Table of DriverExp functions pointers
class _ze_driver_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnRTASFormatCompatibilityCheckExp", c_void_p)                ## _zeDriverRTASFormatCompatibilityCheckExp_t
    ]

###############################################################################
## @brief Function-pointer for zeDeviceGet
if __use_win_types:
    _zeDeviceGet_t = WINFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(c_ulong), POINTER(ze_device_handle_t) )
else:
    _zeDeviceGet_t = CFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(c_ulong), POINTER(ze_device_handle_t) )

###############################################################################
## @brief Function-pointer for zeDeviceGetSubDevices
if __use_win_types:
    _zeDeviceGetSubDevices_t = WINFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(c_ulong), POINTER(ze_device_handle_t) )
else:
    _zeDeviceGetSubDevices_t = CFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(c_ulong), POINTER(ze_device_handle_t) )

###############################################################################
## @brief Function-pointer for zeDeviceGetProperties
if __use_win_types:
    _zeDeviceGetProperties_t = WINFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(ze_device_properties_t) )
else:
    _zeDeviceGetProperties_t = CFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(ze_device_properties_t) )

###############################################################################
## @brief Function-pointer for zeDeviceGetComputeProperties
if __use_win_types:
    _zeDeviceGetComputeProperties_t = WINFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(ze_device_compute_properties_t) )
else:
    _zeDeviceGetComputeProperties_t = CFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(ze_device_compute_properties_t) )

###############################################################################
## @brief Function-pointer for zeDeviceGetModuleProperties
if __use_win_types:
    _zeDeviceGetModuleProperties_t = WINFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(ze_device_module_properties_t) )
else:
    _zeDeviceGetModuleProperties_t = CFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(ze_device_module_properties_t) )

###############################################################################
## @brief Function-pointer for zeDeviceGetCommandQueueGroupProperties
if __use_win_types:
    _zeDeviceGetCommandQueueGroupProperties_t = WINFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(c_ulong), POINTER(ze_command_queue_group_properties_t) )
else:
    _zeDeviceGetCommandQueueGroupProperties_t = CFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(c_ulong), POINTER(ze_command_queue_group_properties_t) )

###############################################################################
## @brief Function-pointer for zeDeviceGetMemoryProperties
if __use_win_types:
    _zeDeviceGetMemoryProperties_t = WINFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(c_ulong), POINTER(ze_device_memory_properties_t) )
else:
    _zeDeviceGetMemoryProperties_t = CFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(c_ulong), POINTER(ze_device_memory_properties_t) )

###############################################################################
## @brief Function-pointer for zeDeviceGetMemoryAccessProperties
if __use_win_types:
    _zeDeviceGetMemoryAccessProperties_t = WINFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(ze_device_memory_access_properties_t) )
else:
    _zeDeviceGetMemoryAccessProperties_t = CFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(ze_device_memory_access_properties_t) )

###############################################################################
## @brief Function-pointer for zeDeviceGetCacheProperties
if __use_win_types:
    _zeDeviceGetCacheProperties_t = WINFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(c_ulong), POINTER(ze_device_cache_properties_t) )
else:
    _zeDeviceGetCacheProperties_t = CFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(c_ulong), POINTER(ze_device_cache_properties_t) )

###############################################################################
## @brief Function-pointer for zeDeviceGetImageProperties
if __use_win_types:
    _zeDeviceGetImageProperties_t = WINFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(ze_device_image_properties_t) )
else:
    _zeDeviceGetImageProperties_t = CFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(ze_device_image_properties_t) )

###############################################################################
## @brief Function-pointer for zeDeviceGetExternalMemoryProperties
if __use_win_types:
    _zeDeviceGetExternalMemoryProperties_t = WINFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(ze_device_external_memory_properties_t) )
else:
    _zeDeviceGetExternalMemoryProperties_t = CFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(ze_device_external_memory_properties_t) )

###############################################################################
## @brief Function-pointer for zeDeviceGetP2PProperties
if __use_win_types:
    _zeDeviceGetP2PProperties_t = WINFUNCTYPE( ze_result_t, ze_device_handle_t, ze_device_handle_t, POINTER(ze_device_p2p_properties_t) )
else:
    _zeDeviceGetP2PProperties_t = CFUNCTYPE( ze_result_t, ze_device_handle_t, ze_device_handle_t, POINTER(ze_device_p2p_properties_t) )

###############################################################################
## @brief Function-pointer for zeDeviceCanAccessPeer
if __use_win_types:
    _zeDeviceCanAccessPeer_t = WINFUNCTYPE( ze_result_t, ze_device_handle_t, ze_device_handle_t, POINTER(ze_bool_t) )
else:
    _zeDeviceCanAccessPeer_t = CFUNCTYPE( ze_result_t, ze_device_handle_t, ze_device_handle_t, POINTER(ze_bool_t) )

###############################################################################
## @brief Function-pointer for zeDeviceGetStatus
if __use_win_types:
    _zeDeviceGetStatus_t = WINFUNCTYPE( ze_result_t, ze_device_handle_t )
else:
    _zeDeviceGetStatus_t = CFUNCTYPE( ze_result_t, ze_device_handle_t )

###############################################################################
## @brief Function-pointer for zeDeviceGetGlobalTimestamps
if __use_win_types:
    _zeDeviceGetGlobalTimestamps_t = WINFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(c_ulonglong), POINTER(c_ulonglong) )
else:
    _zeDeviceGetGlobalTimestamps_t = CFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(c_ulonglong), POINTER(c_ulonglong) )

###############################################################################
## @brief Function-pointer for zeDeviceReserveCacheExt
if __use_win_types:
    _zeDeviceReserveCacheExt_t = WINFUNCTYPE( ze_result_t, ze_device_handle_t, c_size_t, c_size_t )
else:
    _zeDeviceReserveCacheExt_t = CFUNCTYPE( ze_result_t, ze_device_handle_t, c_size_t, c_size_t )

###############################################################################
## @brief Function-pointer for zeDeviceSetCacheAdviceExt
if __use_win_types:
    _zeDeviceSetCacheAdviceExt_t = WINFUNCTYPE( ze_result_t, ze_device_handle_t, c_void_p, c_size_t, ze_cache_ext_region_t )
else:
    _zeDeviceSetCacheAdviceExt_t = CFUNCTYPE( ze_result_t, ze_device_handle_t, c_void_p, c_size_t, ze_cache_ext_region_t )

###############################################################################
## @brief Function-pointer for zeDevicePciGetPropertiesExt
if __use_win_types:
    _zeDevicePciGetPropertiesExt_t = WINFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(ze_pci_ext_properties_t) )
else:
    _zeDevicePciGetPropertiesExt_t = CFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(ze_pci_ext_properties_t) )

###############################################################################
## @brief Function-pointer for zeDeviceGetRootDevice
if __use_win_types:
    _zeDeviceGetRootDevice_t = WINFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(ze_device_handle_t) )
else:
    _zeDeviceGetRootDevice_t = CFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(ze_device_handle_t) )


###############################################################################
## @brief Table of Device functions pointers
class _ze_device_dditable_t(Structure):
    _fields_ = [
        ("pfnGet", c_void_p),                                           ## _zeDeviceGet_t
        ("pfnGetSubDevices", c_void_p),                                 ## _zeDeviceGetSubDevices_t
        ("pfnGetProperties", c_void_p),                                 ## _zeDeviceGetProperties_t
        ("pfnGetComputeProperties", c_void_p),                          ## _zeDeviceGetComputeProperties_t
        ("pfnGetModuleProperties", c_void_p),                           ## _zeDeviceGetModuleProperties_t
        ("pfnGetCommandQueueGroupProperties", c_void_p),                ## _zeDeviceGetCommandQueueGroupProperties_t
        ("pfnGetMemoryProperties", c_void_p),                           ## _zeDeviceGetMemoryProperties_t
        ("pfnGetMemoryAccessProperties", c_void_p),                     ## _zeDeviceGetMemoryAccessProperties_t
        ("pfnGetCacheProperties", c_void_p),                            ## _zeDeviceGetCacheProperties_t
        ("pfnGetImageProperties", c_void_p),                            ## _zeDeviceGetImageProperties_t
        ("pfnGetExternalMemoryProperties", c_void_p),                   ## _zeDeviceGetExternalMemoryProperties_t
        ("pfnGetP2PProperties", c_void_p),                              ## _zeDeviceGetP2PProperties_t
        ("pfnCanAccessPeer", c_void_p),                                 ## _zeDeviceCanAccessPeer_t
        ("pfnGetStatus", c_void_p),                                     ## _zeDeviceGetStatus_t
        ("pfnGetGlobalTimestamps", c_void_p),                           ## _zeDeviceGetGlobalTimestamps_t
        ("pfnReserveCacheExt", c_void_p),                               ## _zeDeviceReserveCacheExt_t
        ("pfnSetCacheAdviceExt", c_void_p),                             ## _zeDeviceSetCacheAdviceExt_t
        ("pfnPciGetPropertiesExt", c_void_p),                           ## _zeDevicePciGetPropertiesExt_t
        ("pfnGetRootDevice", c_void_p)                                  ## _zeDeviceGetRootDevice_t
    ]

###############################################################################
## @brief Function-pointer for zeDeviceGetFabricVertexExp
if __use_win_types:
    _zeDeviceGetFabricVertexExp_t = WINFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(ze_fabric_vertex_handle_t) )
else:
    _zeDeviceGetFabricVertexExp_t = CFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(ze_fabric_vertex_handle_t) )


###############################################################################
## @brief Table of DeviceExp functions pointers
class _ze_device_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnGetFabricVertexExp", c_void_p)                             ## _zeDeviceGetFabricVertexExp_t
    ]

###############################################################################
## @brief Function-pointer for zeContextCreate
if __use_win_types:
    _zeContextCreate_t = WINFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(ze_context_desc_t), POINTER(ze_context_handle_t) )
else:
    _zeContextCreate_t = CFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(ze_context_desc_t), POINTER(ze_context_handle_t) )

###############################################################################
## @brief Function-pointer for zeContextDestroy
if __use_win_types:
    _zeContextDestroy_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t )
else:
    _zeContextDestroy_t = CFUNCTYPE( ze_result_t, ze_context_handle_t )

###############################################################################
## @brief Function-pointer for zeContextGetStatus
if __use_win_types:
    _zeContextGetStatus_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t )
else:
    _zeContextGetStatus_t = CFUNCTYPE( ze_result_t, ze_context_handle_t )

###############################################################################
## @brief Function-pointer for zeContextSystemBarrier
if __use_win_types:
    _zeContextSystemBarrier_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t )
else:
    _zeContextSystemBarrier_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t )

###############################################################################
## @brief Function-pointer for zeContextMakeMemoryResident
if __use_win_types:
    _zeContextMakeMemoryResident_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, c_void_p, c_size_t )
else:
    _zeContextMakeMemoryResident_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, c_void_p, c_size_t )

###############################################################################
## @brief Function-pointer for zeContextEvictMemory
if __use_win_types:
    _zeContextEvictMemory_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, c_void_p, c_size_t )
else:
    _zeContextEvictMemory_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, c_void_p, c_size_t )

###############################################################################
## @brief Function-pointer for zeContextMakeImageResident
if __use_win_types:
    _zeContextMakeImageResident_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, ze_image_handle_t )
else:
    _zeContextMakeImageResident_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, ze_image_handle_t )

###############################################################################
## @brief Function-pointer for zeContextEvictImage
if __use_win_types:
    _zeContextEvictImage_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, ze_image_handle_t )
else:
    _zeContextEvictImage_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, ze_image_handle_t )

###############################################################################
## @brief Function-pointer for zeContextCreateEx
if __use_win_types:
    _zeContextCreateEx_t = WINFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(ze_context_desc_t), c_ulong, POINTER(ze_device_handle_t), POINTER(ze_context_handle_t) )
else:
    _zeContextCreateEx_t = CFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(ze_context_desc_t), c_ulong, POINTER(ze_device_handle_t), POINTER(ze_context_handle_t) )


###############################################################################
## @brief Table of Context functions pointers
class _ze_context_dditable_t(Structure):
    _fields_ = [
        ("pfnCreate", c_void_p),                                        ## _zeContextCreate_t
        ("pfnDestroy", c_void_p),                                       ## _zeContextDestroy_t
        ("pfnGetStatus", c_void_p),                                     ## _zeContextGetStatus_t
        ("pfnSystemBarrier", c_void_p),                                 ## _zeContextSystemBarrier_t
        ("pfnMakeMemoryResident", c_void_p),                            ## _zeContextMakeMemoryResident_t
        ("pfnEvictMemory", c_void_p),                                   ## _zeContextEvictMemory_t
        ("pfnMakeImageResident", c_void_p),                             ## _zeContextMakeImageResident_t
        ("pfnEvictImage", c_void_p),                                    ## _zeContextEvictImage_t
        ("pfnCreateEx", c_void_p)                                       ## _zeContextCreateEx_t
    ]

###############################################################################
## @brief Function-pointer for zeCommandQueueCreate
if __use_win_types:
    _zeCommandQueueCreate_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, POINTER(ze_command_queue_desc_t), POINTER(ze_command_queue_handle_t) )
else:
    _zeCommandQueueCreate_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, POINTER(ze_command_queue_desc_t), POINTER(ze_command_queue_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandQueueDestroy
if __use_win_types:
    _zeCommandQueueDestroy_t = WINFUNCTYPE( ze_result_t, ze_command_queue_handle_t )
else:
    _zeCommandQueueDestroy_t = CFUNCTYPE( ze_result_t, ze_command_queue_handle_t )

###############################################################################
## @brief Function-pointer for zeCommandQueueExecuteCommandLists
if __use_win_types:
    _zeCommandQueueExecuteCommandLists_t = WINFUNCTYPE( ze_result_t, ze_command_queue_handle_t, c_ulong, POINTER(ze_command_list_handle_t), ze_fence_handle_t )
else:
    _zeCommandQueueExecuteCommandLists_t = CFUNCTYPE( ze_result_t, ze_command_queue_handle_t, c_ulong, POINTER(ze_command_list_handle_t), ze_fence_handle_t )

###############################################################################
## @brief Function-pointer for zeCommandQueueSynchronize
if __use_win_types:
    _zeCommandQueueSynchronize_t = WINFUNCTYPE( ze_result_t, ze_command_queue_handle_t, c_ulonglong )
else:
    _zeCommandQueueSynchronize_t = CFUNCTYPE( ze_result_t, ze_command_queue_handle_t, c_ulonglong )

###############################################################################
## @brief Function-pointer for zeCommandQueueGetOrdinal
if __use_win_types:
    _zeCommandQueueGetOrdinal_t = WINFUNCTYPE( ze_result_t, ze_command_queue_handle_t, POINTER(c_ulong) )
else:
    _zeCommandQueueGetOrdinal_t = CFUNCTYPE( ze_result_t, ze_command_queue_handle_t, POINTER(c_ulong) )

###############################################################################
## @brief Function-pointer for zeCommandQueueGetIndex
if __use_win_types:
    _zeCommandQueueGetIndex_t = WINFUNCTYPE( ze_result_t, ze_command_queue_handle_t, POINTER(c_ulong) )
else:
    _zeCommandQueueGetIndex_t = CFUNCTYPE( ze_result_t, ze_command_queue_handle_t, POINTER(c_ulong) )


###############################################################################
## @brief Table of CommandQueue functions pointers
class _ze_command_queue_dditable_t(Structure):
    _fields_ = [
        ("pfnCreate", c_void_p),                                        ## _zeCommandQueueCreate_t
        ("pfnDestroy", c_void_p),                                       ## _zeCommandQueueDestroy_t
        ("pfnExecuteCommandLists", c_void_p),                           ## _zeCommandQueueExecuteCommandLists_t
        ("pfnSynchronize", c_void_p),                                   ## _zeCommandQueueSynchronize_t
        ("pfnGetOrdinal", c_void_p),                                    ## _zeCommandQueueGetOrdinal_t
        ("pfnGetIndex", c_void_p)                                       ## _zeCommandQueueGetIndex_t
    ]

###############################################################################
## @brief Function-pointer for zeCommandListCreate
if __use_win_types:
    _zeCommandListCreate_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, POINTER(ze_command_list_desc_t), POINTER(ze_command_list_handle_t) )
else:
    _zeCommandListCreate_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, POINTER(ze_command_list_desc_t), POINTER(ze_command_list_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListCreateImmediate
if __use_win_types:
    _zeCommandListCreateImmediate_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, POINTER(ze_command_queue_desc_t), POINTER(ze_command_list_handle_t) )
else:
    _zeCommandListCreateImmediate_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, POINTER(ze_command_queue_desc_t), POINTER(ze_command_list_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListDestroy
if __use_win_types:
    _zeCommandListDestroy_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t )
else:
    _zeCommandListDestroy_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t )

###############################################################################
## @brief Function-pointer for zeCommandListClose
if __use_win_types:
    _zeCommandListClose_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t )
else:
    _zeCommandListClose_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t )

###############################################################################
## @brief Function-pointer for zeCommandListReset
if __use_win_types:
    _zeCommandListReset_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t )
else:
    _zeCommandListReset_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t )

###############################################################################
## @brief Function-pointer for zeCommandListAppendWriteGlobalTimestamp
if __use_win_types:
    _zeCommandListAppendWriteGlobalTimestamp_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, POINTER(c_ulonglong), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )
else:
    _zeCommandListAppendWriteGlobalTimestamp_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, POINTER(c_ulonglong), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListAppendBarrier
if __use_win_types:
    _zeCommandListAppendBarrier_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )
else:
    _zeCommandListAppendBarrier_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListAppendMemoryRangesBarrier
if __use_win_types:
    _zeCommandListAppendMemoryRangesBarrier_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_ulong, POINTER(c_size_t), POINTER(c_void_p), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )
else:
    _zeCommandListAppendMemoryRangesBarrier_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_ulong, POINTER(c_size_t), POINTER(c_void_p), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListAppendMemoryCopy
if __use_win_types:
    _zeCommandListAppendMemoryCopy_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_void_p, c_void_p, c_size_t, ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )
else:
    _zeCommandListAppendMemoryCopy_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_void_p, c_void_p, c_size_t, ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListAppendMemoryFill
if __use_win_types:
    _zeCommandListAppendMemoryFill_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_void_p, c_void_p, c_size_t, c_size_t, ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )
else:
    _zeCommandListAppendMemoryFill_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_void_p, c_void_p, c_size_t, c_size_t, ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListAppendMemoryCopyRegion
if __use_win_types:
    _zeCommandListAppendMemoryCopyRegion_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_void_p, POINTER(ze_copy_region_t), c_ulong, c_ulong, c_void_p, POINTER(ze_copy_region_t), c_ulong, c_ulong, ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )
else:
    _zeCommandListAppendMemoryCopyRegion_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_void_p, POINTER(ze_copy_region_t), c_ulong, c_ulong, c_void_p, POINTER(ze_copy_region_t), c_ulong, c_ulong, ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListAppendMemoryCopyFromContext
if __use_win_types:
    _zeCommandListAppendMemoryCopyFromContext_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_void_p, ze_context_handle_t, c_void_p, c_size_t, ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )
else:
    _zeCommandListAppendMemoryCopyFromContext_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_void_p, ze_context_handle_t, c_void_p, c_size_t, ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListAppendImageCopy
if __use_win_types:
    _zeCommandListAppendImageCopy_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_image_handle_t, ze_image_handle_t, ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )
else:
    _zeCommandListAppendImageCopy_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_image_handle_t, ze_image_handle_t, ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListAppendImageCopyRegion
if __use_win_types:
    _zeCommandListAppendImageCopyRegion_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_image_handle_t, ze_image_handle_t, POINTER(ze_image_region_t), POINTER(ze_image_region_t), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )
else:
    _zeCommandListAppendImageCopyRegion_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_image_handle_t, ze_image_handle_t, POINTER(ze_image_region_t), POINTER(ze_image_region_t), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListAppendImageCopyToMemory
if __use_win_types:
    _zeCommandListAppendImageCopyToMemory_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_void_p, ze_image_handle_t, POINTER(ze_image_region_t), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )
else:
    _zeCommandListAppendImageCopyToMemory_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_void_p, ze_image_handle_t, POINTER(ze_image_region_t), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListAppendImageCopyFromMemory
if __use_win_types:
    _zeCommandListAppendImageCopyFromMemory_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_image_handle_t, c_void_p, POINTER(ze_image_region_t), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )
else:
    _zeCommandListAppendImageCopyFromMemory_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_image_handle_t, c_void_p, POINTER(ze_image_region_t), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListAppendMemoryPrefetch
if __use_win_types:
    _zeCommandListAppendMemoryPrefetch_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_void_p, c_size_t )
else:
    _zeCommandListAppendMemoryPrefetch_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_void_p, c_size_t )

###############################################################################
## @brief Function-pointer for zeCommandListAppendMemAdvise
if __use_win_types:
    _zeCommandListAppendMemAdvise_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_device_handle_t, c_void_p, c_size_t, ze_memory_advice_t )
else:
    _zeCommandListAppendMemAdvise_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_device_handle_t, c_void_p, c_size_t, ze_memory_advice_t )

###############################################################################
## @brief Function-pointer for zeCommandListAppendSignalEvent
if __use_win_types:
    _zeCommandListAppendSignalEvent_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_event_handle_t )
else:
    _zeCommandListAppendSignalEvent_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_event_handle_t )

###############################################################################
## @brief Function-pointer for zeCommandListAppendWaitOnEvents
if __use_win_types:
    _zeCommandListAppendWaitOnEvents_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_ulong, POINTER(ze_event_handle_t) )
else:
    _zeCommandListAppendWaitOnEvents_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_ulong, POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListAppendEventReset
if __use_win_types:
    _zeCommandListAppendEventReset_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_event_handle_t )
else:
    _zeCommandListAppendEventReset_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_event_handle_t )

###############################################################################
## @brief Function-pointer for zeCommandListAppendQueryKernelTimestamps
if __use_win_types:
    _zeCommandListAppendQueryKernelTimestamps_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_ulong, POINTER(ze_event_handle_t), c_void_p, POINTER(c_size_t), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )
else:
    _zeCommandListAppendQueryKernelTimestamps_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_ulong, POINTER(ze_event_handle_t), c_void_p, POINTER(c_size_t), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListAppendLaunchKernel
if __use_win_types:
    _zeCommandListAppendLaunchKernel_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_kernel_handle_t, POINTER(ze_group_count_t), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )
else:
    _zeCommandListAppendLaunchKernel_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_kernel_handle_t, POINTER(ze_group_count_t), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListAppendLaunchCooperativeKernel
if __use_win_types:
    _zeCommandListAppendLaunchCooperativeKernel_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_kernel_handle_t, POINTER(ze_group_count_t), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )
else:
    _zeCommandListAppendLaunchCooperativeKernel_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_kernel_handle_t, POINTER(ze_group_count_t), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListAppendLaunchKernelIndirect
if __use_win_types:
    _zeCommandListAppendLaunchKernelIndirect_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_kernel_handle_t, POINTER(ze_group_count_t), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )
else:
    _zeCommandListAppendLaunchKernelIndirect_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_kernel_handle_t, POINTER(ze_group_count_t), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListAppendLaunchMultipleKernelsIndirect
if __use_win_types:
    _zeCommandListAppendLaunchMultipleKernelsIndirect_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_ulong, POINTER(ze_kernel_handle_t), POINTER(c_ulong), POINTER(ze_group_count_t), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )
else:
    _zeCommandListAppendLaunchMultipleKernelsIndirect_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_ulong, POINTER(ze_kernel_handle_t), POINTER(c_ulong), POINTER(ze_group_count_t), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListAppendImageCopyToMemoryExt
if __use_win_types:
    _zeCommandListAppendImageCopyToMemoryExt_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_void_p, ze_image_handle_t, POINTER(ze_image_region_t), c_ulong, c_ulong, ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )
else:
    _zeCommandListAppendImageCopyToMemoryExt_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_void_p, ze_image_handle_t, POINTER(ze_image_region_t), c_ulong, c_ulong, ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListAppendImageCopyFromMemoryExt
if __use_win_types:
    _zeCommandListAppendImageCopyFromMemoryExt_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_image_handle_t, c_void_p, POINTER(ze_image_region_t), c_ulong, c_ulong, ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )
else:
    _zeCommandListAppendImageCopyFromMemoryExt_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, ze_image_handle_t, c_void_p, POINTER(ze_image_region_t), c_ulong, c_ulong, ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListHostSynchronize
if __use_win_types:
    _zeCommandListHostSynchronize_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_ulonglong )
else:
    _zeCommandListHostSynchronize_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_ulonglong )

###############################################################################
## @brief Function-pointer for zeCommandListGetDeviceHandle
if __use_win_types:
    _zeCommandListGetDeviceHandle_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, POINTER(ze_device_handle_t) )
else:
    _zeCommandListGetDeviceHandle_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, POINTER(ze_device_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListGetContextHandle
if __use_win_types:
    _zeCommandListGetContextHandle_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, POINTER(ze_context_handle_t) )
else:
    _zeCommandListGetContextHandle_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, POINTER(ze_context_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListGetOrdinal
if __use_win_types:
    _zeCommandListGetOrdinal_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, POINTER(c_ulong) )
else:
    _zeCommandListGetOrdinal_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, POINTER(c_ulong) )

###############################################################################
## @brief Function-pointer for zeCommandListImmediateGetIndex
if __use_win_types:
    _zeCommandListImmediateGetIndex_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, POINTER(c_ulong) )
else:
    _zeCommandListImmediateGetIndex_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, POINTER(c_ulong) )

###############################################################################
## @brief Function-pointer for zeCommandListIsImmediate
if __use_win_types:
    _zeCommandListIsImmediate_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, POINTER(ze_bool_t) )
else:
    _zeCommandListIsImmediate_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, POINTER(ze_bool_t) )


###############################################################################
## @brief Table of CommandList functions pointers
class _ze_command_list_dditable_t(Structure):
    _fields_ = [
        ("pfnCreate", c_void_p),                                        ## _zeCommandListCreate_t
        ("pfnCreateImmediate", c_void_p),                               ## _zeCommandListCreateImmediate_t
        ("pfnDestroy", c_void_p),                                       ## _zeCommandListDestroy_t
        ("pfnClose", c_void_p),                                         ## _zeCommandListClose_t
        ("pfnReset", c_void_p),                                         ## _zeCommandListReset_t
        ("pfnAppendWriteGlobalTimestamp", c_void_p),                    ## _zeCommandListAppendWriteGlobalTimestamp_t
        ("pfnAppendBarrier", c_void_p),                                 ## _zeCommandListAppendBarrier_t
        ("pfnAppendMemoryRangesBarrier", c_void_p),                     ## _zeCommandListAppendMemoryRangesBarrier_t
        ("pfnAppendMemoryCopy", c_void_p),                              ## _zeCommandListAppendMemoryCopy_t
        ("pfnAppendMemoryFill", c_void_p),                              ## _zeCommandListAppendMemoryFill_t
        ("pfnAppendMemoryCopyRegion", c_void_p),                        ## _zeCommandListAppendMemoryCopyRegion_t
        ("pfnAppendMemoryCopyFromContext", c_void_p),                   ## _zeCommandListAppendMemoryCopyFromContext_t
        ("pfnAppendImageCopy", c_void_p),                               ## _zeCommandListAppendImageCopy_t
        ("pfnAppendImageCopyRegion", c_void_p),                         ## _zeCommandListAppendImageCopyRegion_t
        ("pfnAppendImageCopyToMemory", c_void_p),                       ## _zeCommandListAppendImageCopyToMemory_t
        ("pfnAppendImageCopyFromMemory", c_void_p),                     ## _zeCommandListAppendImageCopyFromMemory_t
        ("pfnAppendMemoryPrefetch", c_void_p),                          ## _zeCommandListAppendMemoryPrefetch_t
        ("pfnAppendMemAdvise", c_void_p),                               ## _zeCommandListAppendMemAdvise_t
        ("pfnAppendSignalEvent", c_void_p),                             ## _zeCommandListAppendSignalEvent_t
        ("pfnAppendWaitOnEvents", c_void_p),                            ## _zeCommandListAppendWaitOnEvents_t
        ("pfnAppendEventReset", c_void_p),                              ## _zeCommandListAppendEventReset_t
        ("pfnAppendQueryKernelTimestamps", c_void_p),                   ## _zeCommandListAppendQueryKernelTimestamps_t
        ("pfnAppendLaunchKernel", c_void_p),                            ## _zeCommandListAppendLaunchKernel_t
        ("pfnAppendLaunchCooperativeKernel", c_void_p),                 ## _zeCommandListAppendLaunchCooperativeKernel_t
        ("pfnAppendLaunchKernelIndirect", c_void_p),                    ## _zeCommandListAppendLaunchKernelIndirect_t
        ("pfnAppendLaunchMultipleKernelsIndirect", c_void_p),           ## _zeCommandListAppendLaunchMultipleKernelsIndirect_t
        ("pfnAppendImageCopyToMemoryExt", c_void_p),                    ## _zeCommandListAppendImageCopyToMemoryExt_t
        ("pfnAppendImageCopyFromMemoryExt", c_void_p),                  ## _zeCommandListAppendImageCopyFromMemoryExt_t
        ("pfnHostSynchronize", c_void_p),                               ## _zeCommandListHostSynchronize_t
        ("pfnGetDeviceHandle", c_void_p),                               ## _zeCommandListGetDeviceHandle_t
        ("pfnGetContextHandle", c_void_p),                              ## _zeCommandListGetContextHandle_t
        ("pfnGetOrdinal", c_void_p),                                    ## _zeCommandListGetOrdinal_t
        ("pfnImmediateGetIndex", c_void_p),                             ## _zeCommandListImmediateGetIndex_t
        ("pfnIsImmediate", c_void_p)                                    ## _zeCommandListIsImmediate_t
    ]

###############################################################################
## @brief Function-pointer for zeCommandListCreateCloneExp
if __use_win_types:
    _zeCommandListCreateCloneExp_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, POINTER(ze_command_list_handle_t) )
else:
    _zeCommandListCreateCloneExp_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, POINTER(ze_command_list_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListImmediateAppendCommandListsExp
if __use_win_types:
    _zeCommandListImmediateAppendCommandListsExp_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_ulong, POINTER(ze_command_list_handle_t), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )
else:
    _zeCommandListImmediateAppendCommandListsExp_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_ulong, POINTER(ze_command_list_handle_t), ze_event_handle_t, c_ulong, POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListGetNextCommandIdExp
if __use_win_types:
    _zeCommandListGetNextCommandIdExp_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, POINTER(ze_mutable_command_id_exp_desc_t), POINTER(c_ulonglong) )
else:
    _zeCommandListGetNextCommandIdExp_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, POINTER(ze_mutable_command_id_exp_desc_t), POINTER(c_ulonglong) )

###############################################################################
## @brief Function-pointer for zeCommandListUpdateMutableCommandsExp
if __use_win_types:
    _zeCommandListUpdateMutableCommandsExp_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, POINTER(ze_mutable_commands_exp_desc_t) )
else:
    _zeCommandListUpdateMutableCommandsExp_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, POINTER(ze_mutable_commands_exp_desc_t) )

###############################################################################
## @brief Function-pointer for zeCommandListUpdateMutableCommandSignalEventExp
if __use_win_types:
    _zeCommandListUpdateMutableCommandSignalEventExp_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_ulonglong, ze_event_handle_t )
else:
    _zeCommandListUpdateMutableCommandSignalEventExp_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_ulonglong, ze_event_handle_t )

###############################################################################
## @brief Function-pointer for zeCommandListUpdateMutableCommandWaitEventsExp
if __use_win_types:
    _zeCommandListUpdateMutableCommandWaitEventsExp_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_ulonglong, c_ulong, POINTER(ze_event_handle_t) )
else:
    _zeCommandListUpdateMutableCommandWaitEventsExp_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_ulonglong, c_ulong, POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeCommandListGetNextCommandIdWithKernelsExp
if __use_win_types:
    _zeCommandListGetNextCommandIdWithKernelsExp_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, POINTER(ze_mutable_command_id_exp_desc_t), c_ulong, POINTER(ze_kernel_handle_t), POINTER(c_ulonglong) )
else:
    _zeCommandListGetNextCommandIdWithKernelsExp_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, POINTER(ze_mutable_command_id_exp_desc_t), c_ulong, POINTER(ze_kernel_handle_t), POINTER(c_ulonglong) )

###############################################################################
## @brief Function-pointer for zeCommandListUpdateMutableCommandKernelsExp
if __use_win_types:
    _zeCommandListUpdateMutableCommandKernelsExp_t = WINFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_ulong, POINTER(c_ulonglong), POINTER(ze_kernel_handle_t) )
else:
    _zeCommandListUpdateMutableCommandKernelsExp_t = CFUNCTYPE( ze_result_t, ze_command_list_handle_t, c_ulong, POINTER(c_ulonglong), POINTER(ze_kernel_handle_t) )


###############################################################################
## @brief Table of CommandListExp functions pointers
class _ze_command_list_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnCreateCloneExp", c_void_p),                                ## _zeCommandListCreateCloneExp_t
        ("pfnImmediateAppendCommandListsExp", c_void_p),                ## _zeCommandListImmediateAppendCommandListsExp_t
        ("pfnGetNextCommandIdExp", c_void_p),                           ## _zeCommandListGetNextCommandIdExp_t
        ("pfnUpdateMutableCommandsExp", c_void_p),                      ## _zeCommandListUpdateMutableCommandsExp_t
        ("pfnUpdateMutableCommandSignalEventExp", c_void_p),            ## _zeCommandListUpdateMutableCommandSignalEventExp_t
        ("pfnUpdateMutableCommandWaitEventsExp", c_void_p),             ## _zeCommandListUpdateMutableCommandWaitEventsExp_t
        ("pfnGetNextCommandIdWithKernelsExp", c_void_p),                ## _zeCommandListGetNextCommandIdWithKernelsExp_t
        ("pfnUpdateMutableCommandKernelsExp", c_void_p)                 ## _zeCommandListUpdateMutableCommandKernelsExp_t
    ]

###############################################################################
## @brief Function-pointer for zeImageGetProperties
if __use_win_types:
    _zeImageGetProperties_t = WINFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(ze_image_desc_t), POINTER(ze_image_properties_t) )
else:
    _zeImageGetProperties_t = CFUNCTYPE( ze_result_t, ze_device_handle_t, POINTER(ze_image_desc_t), POINTER(ze_image_properties_t) )

###############################################################################
## @brief Function-pointer for zeImageCreate
if __use_win_types:
    _zeImageCreate_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, POINTER(ze_image_desc_t), POINTER(ze_image_handle_t) )
else:
    _zeImageCreate_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, POINTER(ze_image_desc_t), POINTER(ze_image_handle_t) )

###############################################################################
## @brief Function-pointer for zeImageDestroy
if __use_win_types:
    _zeImageDestroy_t = WINFUNCTYPE( ze_result_t, ze_image_handle_t )
else:
    _zeImageDestroy_t = CFUNCTYPE( ze_result_t, ze_image_handle_t )

###############################################################################
## @brief Function-pointer for zeImageGetAllocPropertiesExt
if __use_win_types:
    _zeImageGetAllocPropertiesExt_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_image_handle_t, POINTER(ze_image_allocation_ext_properties_t) )
else:
    _zeImageGetAllocPropertiesExt_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_image_handle_t, POINTER(ze_image_allocation_ext_properties_t) )

###############################################################################
## @brief Function-pointer for zeImageViewCreateExt
if __use_win_types:
    _zeImageViewCreateExt_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, POINTER(ze_image_desc_t), ze_image_handle_t, POINTER(ze_image_handle_t) )
else:
    _zeImageViewCreateExt_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, POINTER(ze_image_desc_t), ze_image_handle_t, POINTER(ze_image_handle_t) )


###############################################################################
## @brief Table of Image functions pointers
class _ze_image_dditable_t(Structure):
    _fields_ = [
        ("pfnGetProperties", c_void_p),                                 ## _zeImageGetProperties_t
        ("pfnCreate", c_void_p),                                        ## _zeImageCreate_t
        ("pfnDestroy", c_void_p),                                       ## _zeImageDestroy_t
        ("pfnGetAllocPropertiesExt", c_void_p),                         ## _zeImageGetAllocPropertiesExt_t
        ("pfnViewCreateExt", c_void_p)                                  ## _zeImageViewCreateExt_t
    ]

###############################################################################
## @brief Function-pointer for zeImageGetMemoryPropertiesExp
if __use_win_types:
    _zeImageGetMemoryPropertiesExp_t = WINFUNCTYPE( ze_result_t, ze_image_handle_t, POINTER(ze_image_memory_properties_exp_t) )
else:
    _zeImageGetMemoryPropertiesExp_t = CFUNCTYPE( ze_result_t, ze_image_handle_t, POINTER(ze_image_memory_properties_exp_t) )

###############################################################################
## @brief Function-pointer for zeImageViewCreateExp
if __use_win_types:
    _zeImageViewCreateExp_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, POINTER(ze_image_desc_t), ze_image_handle_t, POINTER(ze_image_handle_t) )
else:
    _zeImageViewCreateExp_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, POINTER(ze_image_desc_t), ze_image_handle_t, POINTER(ze_image_handle_t) )

###############################################################################
## @brief Function-pointer for zeImageGetDeviceOffsetExp
if __use_win_types:
    _zeImageGetDeviceOffsetExp_t = WINFUNCTYPE( ze_result_t, ze_image_handle_t, POINTER(c_ulonglong) )
else:
    _zeImageGetDeviceOffsetExp_t = CFUNCTYPE( ze_result_t, ze_image_handle_t, POINTER(c_ulonglong) )


###############################################################################
## @brief Table of ImageExp functions pointers
class _ze_image_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnGetMemoryPropertiesExp", c_void_p),                        ## _zeImageGetMemoryPropertiesExp_t
        ("pfnViewCreateExp", c_void_p),                                 ## _zeImageViewCreateExp_t
        ("pfnGetDeviceOffsetExp", c_void_p)                             ## _zeImageGetDeviceOffsetExp_t
    ]

###############################################################################
## @brief Function-pointer for zeMemAllocShared
if __use_win_types:
    _zeMemAllocShared_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, POINTER(ze_device_mem_alloc_desc_t), POINTER(ze_host_mem_alloc_desc_t), c_size_t, c_size_t, ze_device_handle_t, POINTER(c_void_p) )
else:
    _zeMemAllocShared_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, POINTER(ze_device_mem_alloc_desc_t), POINTER(ze_host_mem_alloc_desc_t), c_size_t, c_size_t, ze_device_handle_t, POINTER(c_void_p) )

###############################################################################
## @brief Function-pointer for zeMemAllocDevice
if __use_win_types:
    _zeMemAllocDevice_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, POINTER(ze_device_mem_alloc_desc_t), c_size_t, c_size_t, ze_device_handle_t, POINTER(c_void_p) )
else:
    _zeMemAllocDevice_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, POINTER(ze_device_mem_alloc_desc_t), c_size_t, c_size_t, ze_device_handle_t, POINTER(c_void_p) )

###############################################################################
## @brief Function-pointer for zeMemAllocHost
if __use_win_types:
    _zeMemAllocHost_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, POINTER(ze_host_mem_alloc_desc_t), c_size_t, c_size_t, POINTER(c_void_p) )
else:
    _zeMemAllocHost_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, POINTER(ze_host_mem_alloc_desc_t), c_size_t, c_size_t, POINTER(c_void_p) )

###############################################################################
## @brief Function-pointer for zeMemFree
if __use_win_types:
    _zeMemFree_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p )
else:
    _zeMemFree_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p )

###############################################################################
## @brief Function-pointer for zeMemGetAllocProperties
if __use_win_types:
    _zeMemGetAllocProperties_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p, POINTER(ze_memory_allocation_properties_t), POINTER(ze_device_handle_t) )
else:
    _zeMemGetAllocProperties_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p, POINTER(ze_memory_allocation_properties_t), POINTER(ze_device_handle_t) )

###############################################################################
## @brief Function-pointer for zeMemGetAddressRange
if __use_win_types:
    _zeMemGetAddressRange_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p, POINTER(c_void_p), POINTER(c_size_t) )
else:
    _zeMemGetAddressRange_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p, POINTER(c_void_p), POINTER(c_size_t) )

###############################################################################
## @brief Function-pointer for zeMemGetIpcHandle
if __use_win_types:
    _zeMemGetIpcHandle_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p, POINTER(ze_ipc_mem_handle_t) )
else:
    _zeMemGetIpcHandle_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p, POINTER(ze_ipc_mem_handle_t) )

###############################################################################
## @brief Function-pointer for zeMemOpenIpcHandle
if __use_win_types:
    _zeMemOpenIpcHandle_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, ze_ipc_mem_handle_t, ze_ipc_memory_flags_t, POINTER(c_void_p) )
else:
    _zeMemOpenIpcHandle_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, ze_ipc_mem_handle_t, ze_ipc_memory_flags_t, POINTER(c_void_p) )

###############################################################################
## @brief Function-pointer for zeMemCloseIpcHandle
if __use_win_types:
    _zeMemCloseIpcHandle_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p )
else:
    _zeMemCloseIpcHandle_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p )

###############################################################################
## @brief Function-pointer for zeMemFreeExt
if __use_win_types:
    _zeMemFreeExt_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, POINTER(ze_memory_free_ext_desc_t), c_void_p )
else:
    _zeMemFreeExt_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, POINTER(ze_memory_free_ext_desc_t), c_void_p )

###############################################################################
## @brief Function-pointer for zeMemPutIpcHandle
if __use_win_types:
    _zeMemPutIpcHandle_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_ipc_mem_handle_t )
else:
    _zeMemPutIpcHandle_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_ipc_mem_handle_t )

###############################################################################
## @brief Function-pointer for zeMemGetPitchFor2dImage
if __use_win_types:
    _zeMemGetPitchFor2dImage_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, c_size_t, c_size_t, c_int, * )
else:
    _zeMemGetPitchFor2dImage_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, c_size_t, c_size_t, c_int, * )


###############################################################################
## @brief Table of Mem functions pointers
class _ze_mem_dditable_t(Structure):
    _fields_ = [
        ("pfnAllocShared", c_void_p),                                   ## _zeMemAllocShared_t
        ("pfnAllocDevice", c_void_p),                                   ## _zeMemAllocDevice_t
        ("pfnAllocHost", c_void_p),                                     ## _zeMemAllocHost_t
        ("pfnFree", c_void_p),                                          ## _zeMemFree_t
        ("pfnGetAllocProperties", c_void_p),                            ## _zeMemGetAllocProperties_t
        ("pfnGetAddressRange", c_void_p),                               ## _zeMemGetAddressRange_t
        ("pfnGetIpcHandle", c_void_p),                                  ## _zeMemGetIpcHandle_t
        ("pfnOpenIpcHandle", c_void_p),                                 ## _zeMemOpenIpcHandle_t
        ("pfnCloseIpcHandle", c_void_p),                                ## _zeMemCloseIpcHandle_t
        ("pfnFreeExt", c_void_p),                                       ## _zeMemFreeExt_t
        ("pfnPutIpcHandle", c_void_p),                                  ## _zeMemPutIpcHandle_t
        ("pfnGetPitchFor2dImage", c_void_p)                             ## _zeMemGetPitchFor2dImage_t
    ]

###############################################################################
## @brief Function-pointer for zeMemGetIpcHandleFromFileDescriptorExp
if __use_win_types:
    _zeMemGetIpcHandleFromFileDescriptorExp_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, c_ulonglong, POINTER(ze_ipc_mem_handle_t) )
else:
    _zeMemGetIpcHandleFromFileDescriptorExp_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, c_ulonglong, POINTER(ze_ipc_mem_handle_t) )

###############################################################################
## @brief Function-pointer for zeMemGetFileDescriptorFromIpcHandleExp
if __use_win_types:
    _zeMemGetFileDescriptorFromIpcHandleExp_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_ipc_mem_handle_t, POINTER(c_ulonglong) )
else:
    _zeMemGetFileDescriptorFromIpcHandleExp_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_ipc_mem_handle_t, POINTER(c_ulonglong) )

###############################################################################
## @brief Function-pointer for zeMemSetAtomicAccessAttributeExp
if __use_win_types:
    _zeMemSetAtomicAccessAttributeExp_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, c_void_p, c_size_t, ze_memory_atomic_attr_exp_flags_t )
else:
    _zeMemSetAtomicAccessAttributeExp_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, c_void_p, c_size_t, ze_memory_atomic_attr_exp_flags_t )

###############################################################################
## @brief Function-pointer for zeMemGetAtomicAccessAttributeExp
if __use_win_types:
    _zeMemGetAtomicAccessAttributeExp_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, c_void_p, c_size_t, POINTER(ze_memory_atomic_attr_exp_flags_t) )
else:
    _zeMemGetAtomicAccessAttributeExp_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, c_void_p, c_size_t, POINTER(ze_memory_atomic_attr_exp_flags_t) )


###############################################################################
## @brief Table of MemExp functions pointers
class _ze_mem_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnGetIpcHandleFromFileDescriptorExp", c_void_p),             ## _zeMemGetIpcHandleFromFileDescriptorExp_t
        ("pfnGetFileDescriptorFromIpcHandleExp", c_void_p),             ## _zeMemGetFileDescriptorFromIpcHandleExp_t
        ("pfnSetAtomicAccessAttributeExp", c_void_p),                   ## _zeMemSetAtomicAccessAttributeExp_t
        ("pfnGetAtomicAccessAttributeExp", c_void_p)                    ## _zeMemGetAtomicAccessAttributeExp_t
    ]

###############################################################################
## @brief Function-pointer for zeFenceCreate
if __use_win_types:
    _zeFenceCreate_t = WINFUNCTYPE( ze_result_t, ze_command_queue_handle_t, POINTER(ze_fence_desc_t), POINTER(ze_fence_handle_t) )
else:
    _zeFenceCreate_t = CFUNCTYPE( ze_result_t, ze_command_queue_handle_t, POINTER(ze_fence_desc_t), POINTER(ze_fence_handle_t) )

###############################################################################
## @brief Function-pointer for zeFenceDestroy
if __use_win_types:
    _zeFenceDestroy_t = WINFUNCTYPE( ze_result_t, ze_fence_handle_t )
else:
    _zeFenceDestroy_t = CFUNCTYPE( ze_result_t, ze_fence_handle_t )

###############################################################################
## @brief Function-pointer for zeFenceHostSynchronize
if __use_win_types:
    _zeFenceHostSynchronize_t = WINFUNCTYPE( ze_result_t, ze_fence_handle_t, c_ulonglong )
else:
    _zeFenceHostSynchronize_t = CFUNCTYPE( ze_result_t, ze_fence_handle_t, c_ulonglong )

###############################################################################
## @brief Function-pointer for zeFenceQueryStatus
if __use_win_types:
    _zeFenceQueryStatus_t = WINFUNCTYPE( ze_result_t, ze_fence_handle_t )
else:
    _zeFenceQueryStatus_t = CFUNCTYPE( ze_result_t, ze_fence_handle_t )

###############################################################################
## @brief Function-pointer for zeFenceReset
if __use_win_types:
    _zeFenceReset_t = WINFUNCTYPE( ze_result_t, ze_fence_handle_t )
else:
    _zeFenceReset_t = CFUNCTYPE( ze_result_t, ze_fence_handle_t )


###############################################################################
## @brief Table of Fence functions pointers
class _ze_fence_dditable_t(Structure):
    _fields_ = [
        ("pfnCreate", c_void_p),                                        ## _zeFenceCreate_t
        ("pfnDestroy", c_void_p),                                       ## _zeFenceDestroy_t
        ("pfnHostSynchronize", c_void_p),                               ## _zeFenceHostSynchronize_t
        ("pfnQueryStatus", c_void_p),                                   ## _zeFenceQueryStatus_t
        ("pfnReset", c_void_p)                                          ## _zeFenceReset_t
    ]

###############################################################################
## @brief Function-pointer for zeEventPoolCreate
if __use_win_types:
    _zeEventPoolCreate_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, POINTER(ze_event_pool_desc_t), c_ulong, POINTER(ze_device_handle_t), POINTER(ze_event_pool_handle_t) )
else:
    _zeEventPoolCreate_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, POINTER(ze_event_pool_desc_t), c_ulong, POINTER(ze_device_handle_t), POINTER(ze_event_pool_handle_t) )

###############################################################################
## @brief Function-pointer for zeEventPoolDestroy
if __use_win_types:
    _zeEventPoolDestroy_t = WINFUNCTYPE( ze_result_t, ze_event_pool_handle_t )
else:
    _zeEventPoolDestroy_t = CFUNCTYPE( ze_result_t, ze_event_pool_handle_t )

###############################################################################
## @brief Function-pointer for zeEventPoolGetIpcHandle
if __use_win_types:
    _zeEventPoolGetIpcHandle_t = WINFUNCTYPE( ze_result_t, ze_event_pool_handle_t, POINTER(ze_ipc_event_pool_handle_t) )
else:
    _zeEventPoolGetIpcHandle_t = CFUNCTYPE( ze_result_t, ze_event_pool_handle_t, POINTER(ze_ipc_event_pool_handle_t) )

###############################################################################
## @brief Function-pointer for zeEventPoolOpenIpcHandle
if __use_win_types:
    _zeEventPoolOpenIpcHandle_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_ipc_event_pool_handle_t, POINTER(ze_event_pool_handle_t) )
else:
    _zeEventPoolOpenIpcHandle_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_ipc_event_pool_handle_t, POINTER(ze_event_pool_handle_t) )

###############################################################################
## @brief Function-pointer for zeEventPoolCloseIpcHandle
if __use_win_types:
    _zeEventPoolCloseIpcHandle_t = WINFUNCTYPE( ze_result_t, ze_event_pool_handle_t )
else:
    _zeEventPoolCloseIpcHandle_t = CFUNCTYPE( ze_result_t, ze_event_pool_handle_t )

###############################################################################
## @brief Function-pointer for zeEventPoolPutIpcHandle
if __use_win_types:
    _zeEventPoolPutIpcHandle_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_ipc_event_pool_handle_t )
else:
    _zeEventPoolPutIpcHandle_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_ipc_event_pool_handle_t )

###############################################################################
## @brief Function-pointer for zeEventPoolGetContextHandle
if __use_win_types:
    _zeEventPoolGetContextHandle_t = WINFUNCTYPE( ze_result_t, ze_event_pool_handle_t, POINTER(ze_context_handle_t) )
else:
    _zeEventPoolGetContextHandle_t = CFUNCTYPE( ze_result_t, ze_event_pool_handle_t, POINTER(ze_context_handle_t) )

###############################################################################
## @brief Function-pointer for zeEventPoolGetFlags
if __use_win_types:
    _zeEventPoolGetFlags_t = WINFUNCTYPE( ze_result_t, ze_event_pool_handle_t, POINTER(ze_event_pool_flags_t) )
else:
    _zeEventPoolGetFlags_t = CFUNCTYPE( ze_result_t, ze_event_pool_handle_t, POINTER(ze_event_pool_flags_t) )


###############################################################################
## @brief Table of EventPool functions pointers
class _ze_event_pool_dditable_t(Structure):
    _fields_ = [
        ("pfnCreate", c_void_p),                                        ## _zeEventPoolCreate_t
        ("pfnDestroy", c_void_p),                                       ## _zeEventPoolDestroy_t
        ("pfnGetIpcHandle", c_void_p),                                  ## _zeEventPoolGetIpcHandle_t
        ("pfnOpenIpcHandle", c_void_p),                                 ## _zeEventPoolOpenIpcHandle_t
        ("pfnCloseIpcHandle", c_void_p),                                ## _zeEventPoolCloseIpcHandle_t
        ("pfnPutIpcHandle", c_void_p),                                  ## _zeEventPoolPutIpcHandle_t
        ("pfnGetContextHandle", c_void_p),                              ## _zeEventPoolGetContextHandle_t
        ("pfnGetFlags", c_void_p)                                       ## _zeEventPoolGetFlags_t
    ]

###############################################################################
## @brief Function-pointer for zeEventCreate
if __use_win_types:
    _zeEventCreate_t = WINFUNCTYPE( ze_result_t, ze_event_pool_handle_t, POINTER(ze_event_desc_t), POINTER(ze_event_handle_t) )
else:
    _zeEventCreate_t = CFUNCTYPE( ze_result_t, ze_event_pool_handle_t, POINTER(ze_event_desc_t), POINTER(ze_event_handle_t) )

###############################################################################
## @brief Function-pointer for zeEventDestroy
if __use_win_types:
    _zeEventDestroy_t = WINFUNCTYPE( ze_result_t, ze_event_handle_t )
else:
    _zeEventDestroy_t = CFUNCTYPE( ze_result_t, ze_event_handle_t )

###############################################################################
## @brief Function-pointer for zeEventHostSignal
if __use_win_types:
    _zeEventHostSignal_t = WINFUNCTYPE( ze_result_t, ze_event_handle_t )
else:
    _zeEventHostSignal_t = CFUNCTYPE( ze_result_t, ze_event_handle_t )

###############################################################################
## @brief Function-pointer for zeEventHostSynchronize
if __use_win_types:
    _zeEventHostSynchronize_t = WINFUNCTYPE( ze_result_t, ze_event_handle_t, c_ulonglong )
else:
    _zeEventHostSynchronize_t = CFUNCTYPE( ze_result_t, ze_event_handle_t, c_ulonglong )

###############################################################################
## @brief Function-pointer for zeEventQueryStatus
if __use_win_types:
    _zeEventQueryStatus_t = WINFUNCTYPE( ze_result_t, ze_event_handle_t )
else:
    _zeEventQueryStatus_t = CFUNCTYPE( ze_result_t, ze_event_handle_t )

###############################################################################
## @brief Function-pointer for zeEventHostReset
if __use_win_types:
    _zeEventHostReset_t = WINFUNCTYPE( ze_result_t, ze_event_handle_t )
else:
    _zeEventHostReset_t = CFUNCTYPE( ze_result_t, ze_event_handle_t )

###############################################################################
## @brief Function-pointer for zeEventQueryKernelTimestamp
if __use_win_types:
    _zeEventQueryKernelTimestamp_t = WINFUNCTYPE( ze_result_t, ze_event_handle_t, POINTER(ze_kernel_timestamp_result_t) )
else:
    _zeEventQueryKernelTimestamp_t = CFUNCTYPE( ze_result_t, ze_event_handle_t, POINTER(ze_kernel_timestamp_result_t) )

###############################################################################
## @brief Function-pointer for zeEventQueryKernelTimestampsExt
if __use_win_types:
    _zeEventQueryKernelTimestampsExt_t = WINFUNCTYPE( ze_result_t, ze_event_handle_t, ze_device_handle_t, POINTER(c_ulong), POINTER(ze_event_query_kernel_timestamps_results_ext_properties_t) )
else:
    _zeEventQueryKernelTimestampsExt_t = CFUNCTYPE( ze_result_t, ze_event_handle_t, ze_device_handle_t, POINTER(c_ulong), POINTER(ze_event_query_kernel_timestamps_results_ext_properties_t) )

###############################################################################
## @brief Function-pointer for zeEventGetEventPool
if __use_win_types:
    _zeEventGetEventPool_t = WINFUNCTYPE( ze_result_t, ze_event_handle_t, POINTER(ze_event_pool_handle_t) )
else:
    _zeEventGetEventPool_t = CFUNCTYPE( ze_result_t, ze_event_handle_t, POINTER(ze_event_pool_handle_t) )

###############################################################################
## @brief Function-pointer for zeEventGetSignalScope
if __use_win_types:
    _zeEventGetSignalScope_t = WINFUNCTYPE( ze_result_t, ze_event_handle_t, POINTER(ze_event_scope_flags_t) )
else:
    _zeEventGetSignalScope_t = CFUNCTYPE( ze_result_t, ze_event_handle_t, POINTER(ze_event_scope_flags_t) )

###############################################################################
## @brief Function-pointer for zeEventGetWaitScope
if __use_win_types:
    _zeEventGetWaitScope_t = WINFUNCTYPE( ze_result_t, ze_event_handle_t, POINTER(ze_event_scope_flags_t) )
else:
    _zeEventGetWaitScope_t = CFUNCTYPE( ze_result_t, ze_event_handle_t, POINTER(ze_event_scope_flags_t) )


###############################################################################
## @brief Table of Event functions pointers
class _ze_event_dditable_t(Structure):
    _fields_ = [
        ("pfnCreate", c_void_p),                                        ## _zeEventCreate_t
        ("pfnDestroy", c_void_p),                                       ## _zeEventDestroy_t
        ("pfnHostSignal", c_void_p),                                    ## _zeEventHostSignal_t
        ("pfnHostSynchronize", c_void_p),                               ## _zeEventHostSynchronize_t
        ("pfnQueryStatus", c_void_p),                                   ## _zeEventQueryStatus_t
        ("pfnHostReset", c_void_p),                                     ## _zeEventHostReset_t
        ("pfnQueryKernelTimestamp", c_void_p),                          ## _zeEventQueryKernelTimestamp_t
        ("pfnQueryKernelTimestampsExt", c_void_p),                      ## _zeEventQueryKernelTimestampsExt_t
        ("pfnGetEventPool", c_void_p),                                  ## _zeEventGetEventPool_t
        ("pfnGetSignalScope", c_void_p),                                ## _zeEventGetSignalScope_t
        ("pfnGetWaitScope", c_void_p)                                   ## _zeEventGetWaitScope_t
    ]

###############################################################################
## @brief Function-pointer for zeEventQueryTimestampsExp
if __use_win_types:
    _zeEventQueryTimestampsExp_t = WINFUNCTYPE( ze_result_t, ze_event_handle_t, ze_device_handle_t, POINTER(c_ulong), POINTER(ze_kernel_timestamp_result_t) )
else:
    _zeEventQueryTimestampsExp_t = CFUNCTYPE( ze_result_t, ze_event_handle_t, ze_device_handle_t, POINTER(c_ulong), POINTER(ze_kernel_timestamp_result_t) )


###############################################################################
## @brief Table of EventExp functions pointers
class _ze_event_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnQueryTimestampsExp", c_void_p)                             ## _zeEventQueryTimestampsExp_t
    ]

###############################################################################
## @brief Function-pointer for zeModuleCreate
if __use_win_types:
    _zeModuleCreate_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, POINTER(ze_module_desc_t), POINTER(ze_module_handle_t), POINTER(ze_module_build_log_handle_t) )
else:
    _zeModuleCreate_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, POINTER(ze_module_desc_t), POINTER(ze_module_handle_t), POINTER(ze_module_build_log_handle_t) )

###############################################################################
## @brief Function-pointer for zeModuleDestroy
if __use_win_types:
    _zeModuleDestroy_t = WINFUNCTYPE( ze_result_t, ze_module_handle_t )
else:
    _zeModuleDestroy_t = CFUNCTYPE( ze_result_t, ze_module_handle_t )

###############################################################################
## @brief Function-pointer for zeModuleDynamicLink
if __use_win_types:
    _zeModuleDynamicLink_t = WINFUNCTYPE( ze_result_t, c_ulong, POINTER(ze_module_handle_t), POINTER(ze_module_build_log_handle_t) )
else:
    _zeModuleDynamicLink_t = CFUNCTYPE( ze_result_t, c_ulong, POINTER(ze_module_handle_t), POINTER(ze_module_build_log_handle_t) )

###############################################################################
## @brief Function-pointer for zeModuleGetNativeBinary
if __use_win_types:
    _zeModuleGetNativeBinary_t = WINFUNCTYPE( ze_result_t, ze_module_handle_t, POINTER(c_size_t), POINTER(c_ubyte) )
else:
    _zeModuleGetNativeBinary_t = CFUNCTYPE( ze_result_t, ze_module_handle_t, POINTER(c_size_t), POINTER(c_ubyte) )

###############################################################################
## @brief Function-pointer for zeModuleGetGlobalPointer
if __use_win_types:
    _zeModuleGetGlobalPointer_t = WINFUNCTYPE( ze_result_t, ze_module_handle_t, c_char_p, POINTER(c_size_t), POINTER(c_void_p) )
else:
    _zeModuleGetGlobalPointer_t = CFUNCTYPE( ze_result_t, ze_module_handle_t, c_char_p, POINTER(c_size_t), POINTER(c_void_p) )

###############################################################################
## @brief Function-pointer for zeModuleGetKernelNames
if __use_win_types:
    _zeModuleGetKernelNames_t = WINFUNCTYPE( ze_result_t, ze_module_handle_t, POINTER(c_ulong), POINTER(c_char_p) )
else:
    _zeModuleGetKernelNames_t = CFUNCTYPE( ze_result_t, ze_module_handle_t, POINTER(c_ulong), POINTER(c_char_p) )

###############################################################################
## @brief Function-pointer for zeModuleGetProperties
if __use_win_types:
    _zeModuleGetProperties_t = WINFUNCTYPE( ze_result_t, ze_module_handle_t, POINTER(ze_module_properties_t) )
else:
    _zeModuleGetProperties_t = CFUNCTYPE( ze_result_t, ze_module_handle_t, POINTER(ze_module_properties_t) )

###############################################################################
## @brief Function-pointer for zeModuleGetFunctionPointer
if __use_win_types:
    _zeModuleGetFunctionPointer_t = WINFUNCTYPE( ze_result_t, ze_module_handle_t, c_char_p, POINTER(c_void_p) )
else:
    _zeModuleGetFunctionPointer_t = CFUNCTYPE( ze_result_t, ze_module_handle_t, c_char_p, POINTER(c_void_p) )

###############################################################################
## @brief Function-pointer for zeModuleInspectLinkageExt
if __use_win_types:
    _zeModuleInspectLinkageExt_t = WINFUNCTYPE( ze_result_t, POINTER(ze_linkage_inspection_ext_desc_t), c_ulong, POINTER(ze_module_handle_t), POINTER(ze_module_build_log_handle_t) )
else:
    _zeModuleInspectLinkageExt_t = CFUNCTYPE( ze_result_t, POINTER(ze_linkage_inspection_ext_desc_t), c_ulong, POINTER(ze_module_handle_t), POINTER(ze_module_build_log_handle_t) )


###############################################################################
## @brief Table of Module functions pointers
class _ze_module_dditable_t(Structure):
    _fields_ = [
        ("pfnCreate", c_void_p),                                        ## _zeModuleCreate_t
        ("pfnDestroy", c_void_p),                                       ## _zeModuleDestroy_t
        ("pfnDynamicLink", c_void_p),                                   ## _zeModuleDynamicLink_t
        ("pfnGetNativeBinary", c_void_p),                               ## _zeModuleGetNativeBinary_t
        ("pfnGetGlobalPointer", c_void_p),                              ## _zeModuleGetGlobalPointer_t
        ("pfnGetKernelNames", c_void_p),                                ## _zeModuleGetKernelNames_t
        ("pfnGetProperties", c_void_p),                                 ## _zeModuleGetProperties_t
        ("pfnGetFunctionPointer", c_void_p),                            ## _zeModuleGetFunctionPointer_t
        ("pfnInspectLinkageExt", c_void_p)                              ## _zeModuleInspectLinkageExt_t
    ]

###############################################################################
## @brief Function-pointer for zeModuleBuildLogDestroy
if __use_win_types:
    _zeModuleBuildLogDestroy_t = WINFUNCTYPE( ze_result_t, ze_module_build_log_handle_t )
else:
    _zeModuleBuildLogDestroy_t = CFUNCTYPE( ze_result_t, ze_module_build_log_handle_t )

###############################################################################
## @brief Function-pointer for zeModuleBuildLogGetString
if __use_win_types:
    _zeModuleBuildLogGetString_t = WINFUNCTYPE( ze_result_t, ze_module_build_log_handle_t, POINTER(c_size_t), c_char_p )
else:
    _zeModuleBuildLogGetString_t = CFUNCTYPE( ze_result_t, ze_module_build_log_handle_t, POINTER(c_size_t), c_char_p )


###############################################################################
## @brief Table of ModuleBuildLog functions pointers
class _ze_module_build_log_dditable_t(Structure):
    _fields_ = [
        ("pfnDestroy", c_void_p),                                       ## _zeModuleBuildLogDestroy_t
        ("pfnGetString", c_void_p)                                      ## _zeModuleBuildLogGetString_t
    ]

###############################################################################
## @brief Function-pointer for zeKernelCreate
if __use_win_types:
    _zeKernelCreate_t = WINFUNCTYPE( ze_result_t, ze_module_handle_t, POINTER(ze_kernel_desc_t), POINTER(ze_kernel_handle_t) )
else:
    _zeKernelCreate_t = CFUNCTYPE( ze_result_t, ze_module_handle_t, POINTER(ze_kernel_desc_t), POINTER(ze_kernel_handle_t) )

###############################################################################
## @brief Function-pointer for zeKernelDestroy
if __use_win_types:
    _zeKernelDestroy_t = WINFUNCTYPE( ze_result_t, ze_kernel_handle_t )
else:
    _zeKernelDestroy_t = CFUNCTYPE( ze_result_t, ze_kernel_handle_t )

###############################################################################
## @brief Function-pointer for zeKernelSetCacheConfig
if __use_win_types:
    _zeKernelSetCacheConfig_t = WINFUNCTYPE( ze_result_t, ze_kernel_handle_t, ze_cache_config_flags_t )
else:
    _zeKernelSetCacheConfig_t = CFUNCTYPE( ze_result_t, ze_kernel_handle_t, ze_cache_config_flags_t )

###############################################################################
## @brief Function-pointer for zeKernelSetGroupSize
if __use_win_types:
    _zeKernelSetGroupSize_t = WINFUNCTYPE( ze_result_t, ze_kernel_handle_t, c_ulong, c_ulong, c_ulong )
else:
    _zeKernelSetGroupSize_t = CFUNCTYPE( ze_result_t, ze_kernel_handle_t, c_ulong, c_ulong, c_ulong )

###############################################################################
## @brief Function-pointer for zeKernelSuggestGroupSize
if __use_win_types:
    _zeKernelSuggestGroupSize_t = WINFUNCTYPE( ze_result_t, ze_kernel_handle_t, c_ulong, c_ulong, c_ulong, POINTER(c_ulong), POINTER(c_ulong), POINTER(c_ulong) )
else:
    _zeKernelSuggestGroupSize_t = CFUNCTYPE( ze_result_t, ze_kernel_handle_t, c_ulong, c_ulong, c_ulong, POINTER(c_ulong), POINTER(c_ulong), POINTER(c_ulong) )

###############################################################################
## @brief Function-pointer for zeKernelSuggestMaxCooperativeGroupCount
if __use_win_types:
    _zeKernelSuggestMaxCooperativeGroupCount_t = WINFUNCTYPE( ze_result_t, ze_kernel_handle_t, POINTER(c_ulong) )
else:
    _zeKernelSuggestMaxCooperativeGroupCount_t = CFUNCTYPE( ze_result_t, ze_kernel_handle_t, POINTER(c_ulong) )

###############################################################################
## @brief Function-pointer for zeKernelSetArgumentValue
if __use_win_types:
    _zeKernelSetArgumentValue_t = WINFUNCTYPE( ze_result_t, ze_kernel_handle_t, c_ulong, c_size_t, c_void_p )
else:
    _zeKernelSetArgumentValue_t = CFUNCTYPE( ze_result_t, ze_kernel_handle_t, c_ulong, c_size_t, c_void_p )

###############################################################################
## @brief Function-pointer for zeKernelSetIndirectAccess
if __use_win_types:
    _zeKernelSetIndirectAccess_t = WINFUNCTYPE( ze_result_t, ze_kernel_handle_t, ze_kernel_indirect_access_flags_t )
else:
    _zeKernelSetIndirectAccess_t = CFUNCTYPE( ze_result_t, ze_kernel_handle_t, ze_kernel_indirect_access_flags_t )

###############################################################################
## @brief Function-pointer for zeKernelGetIndirectAccess
if __use_win_types:
    _zeKernelGetIndirectAccess_t = WINFUNCTYPE( ze_result_t, ze_kernel_handle_t, POINTER(ze_kernel_indirect_access_flags_t) )
else:
    _zeKernelGetIndirectAccess_t = CFUNCTYPE( ze_result_t, ze_kernel_handle_t, POINTER(ze_kernel_indirect_access_flags_t) )

###############################################################################
## @brief Function-pointer for zeKernelGetSourceAttributes
if __use_win_types:
    _zeKernelGetSourceAttributes_t = WINFUNCTYPE( ze_result_t, ze_kernel_handle_t, POINTER(c_ulong), POINTER(c_char_p) )
else:
    _zeKernelGetSourceAttributes_t = CFUNCTYPE( ze_result_t, ze_kernel_handle_t, POINTER(c_ulong), POINTER(c_char_p) )

###############################################################################
## @brief Function-pointer for zeKernelGetProperties
if __use_win_types:
    _zeKernelGetProperties_t = WINFUNCTYPE( ze_result_t, ze_kernel_handle_t, POINTER(ze_kernel_properties_t) )
else:
    _zeKernelGetProperties_t = CFUNCTYPE( ze_result_t, ze_kernel_handle_t, POINTER(ze_kernel_properties_t) )

###############################################################################
## @brief Function-pointer for zeKernelGetName
if __use_win_types:
    _zeKernelGetName_t = WINFUNCTYPE( ze_result_t, ze_kernel_handle_t, POINTER(c_size_t), c_char_p )
else:
    _zeKernelGetName_t = CFUNCTYPE( ze_result_t, ze_kernel_handle_t, POINTER(c_size_t), c_char_p )


###############################################################################
## @brief Table of Kernel functions pointers
class _ze_kernel_dditable_t(Structure):
    _fields_ = [
        ("pfnCreate", c_void_p),                                        ## _zeKernelCreate_t
        ("pfnDestroy", c_void_p),                                       ## _zeKernelDestroy_t
        ("pfnSetCacheConfig", c_void_p),                                ## _zeKernelSetCacheConfig_t
        ("pfnSetGroupSize", c_void_p),                                  ## _zeKernelSetGroupSize_t
        ("pfnSuggestGroupSize", c_void_p),                              ## _zeKernelSuggestGroupSize_t
        ("pfnSuggestMaxCooperativeGroupCount", c_void_p),               ## _zeKernelSuggestMaxCooperativeGroupCount_t
        ("pfnSetArgumentValue", c_void_p),                              ## _zeKernelSetArgumentValue_t
        ("pfnSetIndirectAccess", c_void_p),                             ## _zeKernelSetIndirectAccess_t
        ("pfnGetIndirectAccess", c_void_p),                             ## _zeKernelGetIndirectAccess_t
        ("pfnGetSourceAttributes", c_void_p),                           ## _zeKernelGetSourceAttributes_t
        ("pfnGetProperties", c_void_p),                                 ## _zeKernelGetProperties_t
        ("pfnGetName", c_void_p)                                        ## _zeKernelGetName_t
    ]

###############################################################################
## @brief Function-pointer for zeKernelSetGlobalOffsetExp
if __use_win_types:
    _zeKernelSetGlobalOffsetExp_t = WINFUNCTYPE( ze_result_t, ze_kernel_handle_t, c_ulong, c_ulong, c_ulong )
else:
    _zeKernelSetGlobalOffsetExp_t = CFUNCTYPE( ze_result_t, ze_kernel_handle_t, c_ulong, c_ulong, c_ulong )

###############################################################################
## @brief Function-pointer for zeKernelSchedulingHintExp
if __use_win_types:
    _zeKernelSchedulingHintExp_t = WINFUNCTYPE( ze_result_t, ze_kernel_handle_t, POINTER(ze_scheduling_hint_exp_desc_t) )
else:
    _zeKernelSchedulingHintExp_t = CFUNCTYPE( ze_result_t, ze_kernel_handle_t, POINTER(ze_scheduling_hint_exp_desc_t) )

###############################################################################
## @brief Function-pointer for zeKernelGetBinaryExp
if __use_win_types:
    _zeKernelGetBinaryExp_t = WINFUNCTYPE( ze_result_t, ze_kernel_handle_t, POINTER(c_size_t), POINTER(c_ubyte) )
else:
    _zeKernelGetBinaryExp_t = CFUNCTYPE( ze_result_t, ze_kernel_handle_t, POINTER(c_size_t), POINTER(c_ubyte) )


###############################################################################
## @brief Table of KernelExp functions pointers
class _ze_kernel_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnSetGlobalOffsetExp", c_void_p),                            ## _zeKernelSetGlobalOffsetExp_t
        ("pfnSchedulingHintExp", c_void_p),                             ## _zeKernelSchedulingHintExp_t
        ("pfnGetBinaryExp", c_void_p)                                   ## _zeKernelGetBinaryExp_t
    ]

###############################################################################
## @brief Function-pointer for zeSamplerCreate
if __use_win_types:
    _zeSamplerCreate_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, POINTER(ze_sampler_desc_t), POINTER(ze_sampler_handle_t) )
else:
    _zeSamplerCreate_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, POINTER(ze_sampler_desc_t), POINTER(ze_sampler_handle_t) )

###############################################################################
## @brief Function-pointer for zeSamplerDestroy
if __use_win_types:
    _zeSamplerDestroy_t = WINFUNCTYPE( ze_result_t, ze_sampler_handle_t )
else:
    _zeSamplerDestroy_t = CFUNCTYPE( ze_result_t, ze_sampler_handle_t )


###############################################################################
## @brief Table of Sampler functions pointers
class _ze_sampler_dditable_t(Structure):
    _fields_ = [
        ("pfnCreate", c_void_p),                                        ## _zeSamplerCreate_t
        ("pfnDestroy", c_void_p)                                        ## _zeSamplerDestroy_t
    ]

###############################################################################
## @brief Function-pointer for zePhysicalMemCreate
if __use_win_types:
    _zePhysicalMemCreate_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, POINTER(ze_physical_mem_desc_t), POINTER(ze_physical_mem_handle_t) )
else:
    _zePhysicalMemCreate_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, POINTER(ze_physical_mem_desc_t), POINTER(ze_physical_mem_handle_t) )

###############################################################################
## @brief Function-pointer for zePhysicalMemDestroy
if __use_win_types:
    _zePhysicalMemDestroy_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_physical_mem_handle_t )
else:
    _zePhysicalMemDestroy_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_physical_mem_handle_t )


###############################################################################
## @brief Table of PhysicalMem functions pointers
class _ze_physical_mem_dditable_t(Structure):
    _fields_ = [
        ("pfnCreate", c_void_p),                                        ## _zePhysicalMemCreate_t
        ("pfnDestroy", c_void_p)                                        ## _zePhysicalMemDestroy_t
    ]

###############################################################################
## @brief Function-pointer for zeVirtualMemReserve
if __use_win_types:
    _zeVirtualMemReserve_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p, c_size_t, POINTER(c_void_p) )
else:
    _zeVirtualMemReserve_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p, c_size_t, POINTER(c_void_p) )

###############################################################################
## @brief Function-pointer for zeVirtualMemFree
if __use_win_types:
    _zeVirtualMemFree_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p, c_size_t )
else:
    _zeVirtualMemFree_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p, c_size_t )

###############################################################################
## @brief Function-pointer for zeVirtualMemQueryPageSize
if __use_win_types:
    _zeVirtualMemQueryPageSize_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, c_size_t, POINTER(c_size_t) )
else:
    _zeVirtualMemQueryPageSize_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, ze_device_handle_t, c_size_t, POINTER(c_size_t) )

###############################################################################
## @brief Function-pointer for zeVirtualMemMap
if __use_win_types:
    _zeVirtualMemMap_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p, c_size_t, ze_physical_mem_handle_t, c_size_t, ze_memory_access_attribute_t )
else:
    _zeVirtualMemMap_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p, c_size_t, ze_physical_mem_handle_t, c_size_t, ze_memory_access_attribute_t )

###############################################################################
## @brief Function-pointer for zeVirtualMemUnmap
if __use_win_types:
    _zeVirtualMemUnmap_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p, c_size_t )
else:
    _zeVirtualMemUnmap_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p, c_size_t )

###############################################################################
## @brief Function-pointer for zeVirtualMemSetAccessAttribute
if __use_win_types:
    _zeVirtualMemSetAccessAttribute_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p, c_size_t, ze_memory_access_attribute_t )
else:
    _zeVirtualMemSetAccessAttribute_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p, c_size_t, ze_memory_access_attribute_t )

###############################################################################
## @brief Function-pointer for zeVirtualMemGetAccessAttribute
if __use_win_types:
    _zeVirtualMemGetAccessAttribute_t = WINFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p, c_size_t, POINTER(ze_memory_access_attribute_t), POINTER(c_size_t) )
else:
    _zeVirtualMemGetAccessAttribute_t = CFUNCTYPE( ze_result_t, ze_context_handle_t, c_void_p, c_size_t, POINTER(ze_memory_access_attribute_t), POINTER(c_size_t) )


###############################################################################
## @brief Table of VirtualMem functions pointers
class _ze_virtual_mem_dditable_t(Structure):
    _fields_ = [
        ("pfnReserve", c_void_p),                                       ## _zeVirtualMemReserve_t
        ("pfnFree", c_void_p),                                          ## _zeVirtualMemFree_t
        ("pfnQueryPageSize", c_void_p),                                 ## _zeVirtualMemQueryPageSize_t
        ("pfnMap", c_void_p),                                           ## _zeVirtualMemMap_t
        ("pfnUnmap", c_void_p),                                         ## _zeVirtualMemUnmap_t
        ("pfnSetAccessAttribute", c_void_p),                            ## _zeVirtualMemSetAccessAttribute_t
        ("pfnGetAccessAttribute", c_void_p)                             ## _zeVirtualMemGetAccessAttribute_t
    ]

###############################################################################
## @brief Function-pointer for zeFabricVertexGetExp
if __use_win_types:
    _zeFabricVertexGetExp_t = WINFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(c_ulong), POINTER(ze_fabric_vertex_handle_t) )
else:
    _zeFabricVertexGetExp_t = CFUNCTYPE( ze_result_t, ze_driver_handle_t, POINTER(c_ulong), POINTER(ze_fabric_vertex_handle_t) )

###############################################################################
## @brief Function-pointer for zeFabricVertexGetSubVerticesExp
if __use_win_types:
    _zeFabricVertexGetSubVerticesExp_t = WINFUNCTYPE( ze_result_t, ze_fabric_vertex_handle_t, POINTER(c_ulong), POINTER(ze_fabric_vertex_handle_t) )
else:
    _zeFabricVertexGetSubVerticesExp_t = CFUNCTYPE( ze_result_t, ze_fabric_vertex_handle_t, POINTER(c_ulong), POINTER(ze_fabric_vertex_handle_t) )

###############################################################################
## @brief Function-pointer for zeFabricVertexGetPropertiesExp
if __use_win_types:
    _zeFabricVertexGetPropertiesExp_t = WINFUNCTYPE( ze_result_t, ze_fabric_vertex_handle_t, POINTER(ze_fabric_vertex_exp_properties_t) )
else:
    _zeFabricVertexGetPropertiesExp_t = CFUNCTYPE( ze_result_t, ze_fabric_vertex_handle_t, POINTER(ze_fabric_vertex_exp_properties_t) )

###############################################################################
## @brief Function-pointer for zeFabricVertexGetDeviceExp
if __use_win_types:
    _zeFabricVertexGetDeviceExp_t = WINFUNCTYPE( ze_result_t, ze_fabric_vertex_handle_t, POINTER(ze_device_handle_t) )
else:
    _zeFabricVertexGetDeviceExp_t = CFUNCTYPE( ze_result_t, ze_fabric_vertex_handle_t, POINTER(ze_device_handle_t) )


###############################################################################
## @brief Table of FabricVertexExp functions pointers
class _ze_fabric_vertex_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnGetExp", c_void_p),                                        ## _zeFabricVertexGetExp_t
        ("pfnGetSubVerticesExp", c_void_p),                             ## _zeFabricVertexGetSubVerticesExp_t
        ("pfnGetPropertiesExp", c_void_p),                              ## _zeFabricVertexGetPropertiesExp_t
        ("pfnGetDeviceExp", c_void_p)                                   ## _zeFabricVertexGetDeviceExp_t
    ]

###############################################################################
## @brief Function-pointer for zeFabricEdgeGetExp
if __use_win_types:
    _zeFabricEdgeGetExp_t = WINFUNCTYPE( ze_result_t, ze_fabric_vertex_handle_t, ze_fabric_vertex_handle_t, POINTER(c_ulong), POINTER(ze_fabric_edge_handle_t) )
else:
    _zeFabricEdgeGetExp_t = CFUNCTYPE( ze_result_t, ze_fabric_vertex_handle_t, ze_fabric_vertex_handle_t, POINTER(c_ulong), POINTER(ze_fabric_edge_handle_t) )

###############################################################################
## @brief Function-pointer for zeFabricEdgeGetVerticesExp
if __use_win_types:
    _zeFabricEdgeGetVerticesExp_t = WINFUNCTYPE( ze_result_t, ze_fabric_edge_handle_t, POINTER(ze_fabric_vertex_handle_t), POINTER(ze_fabric_vertex_handle_t) )
else:
    _zeFabricEdgeGetVerticesExp_t = CFUNCTYPE( ze_result_t, ze_fabric_edge_handle_t, POINTER(ze_fabric_vertex_handle_t), POINTER(ze_fabric_vertex_handle_t) )

###############################################################################
## @brief Function-pointer for zeFabricEdgeGetPropertiesExp
if __use_win_types:
    _zeFabricEdgeGetPropertiesExp_t = WINFUNCTYPE( ze_result_t, ze_fabric_edge_handle_t, POINTER(ze_fabric_edge_exp_properties_t) )
else:
    _zeFabricEdgeGetPropertiesExp_t = CFUNCTYPE( ze_result_t, ze_fabric_edge_handle_t, POINTER(ze_fabric_edge_exp_properties_t) )


###############################################################################
## @brief Table of FabricEdgeExp functions pointers
class _ze_fabric_edge_exp_dditable_t(Structure):
    _fields_ = [
        ("pfnGetExp", c_void_p),                                        ## _zeFabricEdgeGetExp_t
        ("pfnGetVerticesExp", c_void_p),                                ## _zeFabricEdgeGetVerticesExp_t
        ("pfnGetPropertiesExp", c_void_p)                               ## _zeFabricEdgeGetPropertiesExp_t
    ]

###############################################################################
class _ze_dditable_t(Structure):
    _fields_ = [
        ("RTASBuilderExp", _ze_rtas_builder_exp_dditable_t),
        ("RTASParallelOperationExp", _ze_rtas_parallel_operation_exp_dditable_t),
        ("Global", _ze_global_dditable_t),
        ("Driver", _ze_driver_dditable_t),
        ("DriverExp", _ze_driver_exp_dditable_t),
        ("Device", _ze_device_dditable_t),
        ("DeviceExp", _ze_device_exp_dditable_t),
        ("Context", _ze_context_dditable_t),
        ("CommandQueue", _ze_command_queue_dditable_t),
        ("CommandList", _ze_command_list_dditable_t),
        ("CommandListExp", _ze_command_list_exp_dditable_t),
        ("Image", _ze_image_dditable_t),
        ("ImageExp", _ze_image_exp_dditable_t),
        ("Mem", _ze_mem_dditable_t),
        ("MemExp", _ze_mem_exp_dditable_t),
        ("Fence", _ze_fence_dditable_t),
        ("EventPool", _ze_event_pool_dditable_t),
        ("Event", _ze_event_dditable_t),
        ("EventExp", _ze_event_exp_dditable_t),
        ("Module", _ze_module_dditable_t),
        ("ModuleBuildLog", _ze_module_build_log_dditable_t),
        ("Kernel", _ze_kernel_dditable_t),
        ("KernelExp", _ze_kernel_exp_dditable_t),
        ("Sampler", _ze_sampler_dditable_t),
        ("PhysicalMem", _ze_physical_mem_dditable_t),
        ("VirtualMem", _ze_virtual_mem_dditable_t),
        ("FabricVertexExp", _ze_fabric_vertex_exp_dditable_t),
        ("FabricEdgeExp", _ze_fabric_edge_exp_dditable_t)
    ]

###############################################################################
## @brief ze device-driver interfaces
class ZE_DDI:
    def __init__(self, version : ze_api_version_t):
        # load the ze_loader library
        if "Windows" == platform.uname()[0]:
            self.__dll = WinDLL("ze_loader.dll")
        else:
            self.__dll = CDLL("ze_loader.so")

        # fill the ddi tables
        self.__dditable = _ze_dditable_t()

        # call driver to get function pointers
        _RTASBuilderExp = _ze_rtas_builder_exp_dditable_t()
        r = ze_result_v(self.__dll.zeGetRTASBuilderExpProcAddrTable(version, byref(_RTASBuilderExp)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.RTASBuilderExp = _RTASBuilderExp

        # attach function interface to function address
        self.zeRTASBuilderCreateExp = _zeRTASBuilderCreateExp_t(self.__dditable.RTASBuilderExp.pfnCreateExp)
        self.zeRTASBuilderGetBuildPropertiesExp = _zeRTASBuilderGetBuildPropertiesExp_t(self.__dditable.RTASBuilderExp.pfnGetBuildPropertiesExp)
        self.zeRTASBuilderBuildExp = _zeRTASBuilderBuildExp_t(self.__dditable.RTASBuilderExp.pfnBuildExp)
        self.zeRTASBuilderDestroyExp = _zeRTASBuilderDestroyExp_t(self.__dditable.RTASBuilderExp.pfnDestroyExp)

        # call driver to get function pointers
        _RTASParallelOperationExp = _ze_rtas_parallel_operation_exp_dditable_t()
        r = ze_result_v(self.__dll.zeGetRTASParallelOperationExpProcAddrTable(version, byref(_RTASParallelOperationExp)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.RTASParallelOperationExp = _RTASParallelOperationExp

        # attach function interface to function address
        self.zeRTASParallelOperationCreateExp = _zeRTASParallelOperationCreateExp_t(self.__dditable.RTASParallelOperationExp.pfnCreateExp)
        self.zeRTASParallelOperationGetPropertiesExp = _zeRTASParallelOperationGetPropertiesExp_t(self.__dditable.RTASParallelOperationExp.pfnGetPropertiesExp)
        self.zeRTASParallelOperationJoinExp = _zeRTASParallelOperationJoinExp_t(self.__dditable.RTASParallelOperationExp.pfnJoinExp)
        self.zeRTASParallelOperationDestroyExp = _zeRTASParallelOperationDestroyExp_t(self.__dditable.RTASParallelOperationExp.pfnDestroyExp)

        # call driver to get function pointers
        _Global = _ze_global_dditable_t()
        r = ze_result_v(self.__dll.zeGetGlobalProcAddrTable(version, byref(_Global)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Global = _Global

        # attach function interface to function address
        self.zeInit = _zeInit_t(self.__dditable.Global.pfnInit)
        self.zeInitDrivers = _zeInitDrivers_t(self.__dditable.Global.pfnInitDrivers)

        # call driver to get function pointers
        _Driver = _ze_driver_dditable_t()
        r = ze_result_v(self.__dll.zeGetDriverProcAddrTable(version, byref(_Driver)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Driver = _Driver

        # attach function interface to function address
        self.zeDriverGet = _zeDriverGet_t(self.__dditable.Driver.pfnGet)
        self.zeDriverGetApiVersion = _zeDriverGetApiVersion_t(self.__dditable.Driver.pfnGetApiVersion)
        self.zeDriverGetProperties = _zeDriverGetProperties_t(self.__dditable.Driver.pfnGetProperties)
        self.zeDriverGetIpcProperties = _zeDriverGetIpcProperties_t(self.__dditable.Driver.pfnGetIpcProperties)
        self.zeDriverGetExtensionProperties = _zeDriverGetExtensionProperties_t(self.__dditable.Driver.pfnGetExtensionProperties)
        self.zeDriverGetExtensionFunctionAddress = _zeDriverGetExtensionFunctionAddress_t(self.__dditable.Driver.pfnGetExtensionFunctionAddress)
        self.zeDriverGetLastErrorDescription = _zeDriverGetLastErrorDescription_t(self.__dditable.Driver.pfnGetLastErrorDescription)

        # call driver to get function pointers
        _DriverExp = _ze_driver_exp_dditable_t()
        r = ze_result_v(self.__dll.zeGetDriverExpProcAddrTable(version, byref(_DriverExp)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.DriverExp = _DriverExp

        # attach function interface to function address
        self.zeDriverRTASFormatCompatibilityCheckExp = _zeDriverRTASFormatCompatibilityCheckExp_t(self.__dditable.DriverExp.pfnRTASFormatCompatibilityCheckExp)

        # call driver to get function pointers
        _Device = _ze_device_dditable_t()
        r = ze_result_v(self.__dll.zeGetDeviceProcAddrTable(version, byref(_Device)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Device = _Device

        # attach function interface to function address
        self.zeDeviceGet = _zeDeviceGet_t(self.__dditable.Device.pfnGet)
        self.zeDeviceGetSubDevices = _zeDeviceGetSubDevices_t(self.__dditable.Device.pfnGetSubDevices)
        self.zeDeviceGetProperties = _zeDeviceGetProperties_t(self.__dditable.Device.pfnGetProperties)
        self.zeDeviceGetComputeProperties = _zeDeviceGetComputeProperties_t(self.__dditable.Device.pfnGetComputeProperties)
        self.zeDeviceGetModuleProperties = _zeDeviceGetModuleProperties_t(self.__dditable.Device.pfnGetModuleProperties)
        self.zeDeviceGetCommandQueueGroupProperties = _zeDeviceGetCommandQueueGroupProperties_t(self.__dditable.Device.pfnGetCommandQueueGroupProperties)
        self.zeDeviceGetMemoryProperties = _zeDeviceGetMemoryProperties_t(self.__dditable.Device.pfnGetMemoryProperties)
        self.zeDeviceGetMemoryAccessProperties = _zeDeviceGetMemoryAccessProperties_t(self.__dditable.Device.pfnGetMemoryAccessProperties)
        self.zeDeviceGetCacheProperties = _zeDeviceGetCacheProperties_t(self.__dditable.Device.pfnGetCacheProperties)
        self.zeDeviceGetImageProperties = _zeDeviceGetImageProperties_t(self.__dditable.Device.pfnGetImageProperties)
        self.zeDeviceGetExternalMemoryProperties = _zeDeviceGetExternalMemoryProperties_t(self.__dditable.Device.pfnGetExternalMemoryProperties)
        self.zeDeviceGetP2PProperties = _zeDeviceGetP2PProperties_t(self.__dditable.Device.pfnGetP2PProperties)
        self.zeDeviceCanAccessPeer = _zeDeviceCanAccessPeer_t(self.__dditable.Device.pfnCanAccessPeer)
        self.zeDeviceGetStatus = _zeDeviceGetStatus_t(self.__dditable.Device.pfnGetStatus)
        self.zeDeviceGetGlobalTimestamps = _zeDeviceGetGlobalTimestamps_t(self.__dditable.Device.pfnGetGlobalTimestamps)
        self.zeDeviceReserveCacheExt = _zeDeviceReserveCacheExt_t(self.__dditable.Device.pfnReserveCacheExt)
        self.zeDeviceSetCacheAdviceExt = _zeDeviceSetCacheAdviceExt_t(self.__dditable.Device.pfnSetCacheAdviceExt)
        self.zeDevicePciGetPropertiesExt = _zeDevicePciGetPropertiesExt_t(self.__dditable.Device.pfnPciGetPropertiesExt)
        self.zeDeviceGetRootDevice = _zeDeviceGetRootDevice_t(self.__dditable.Device.pfnGetRootDevice)

        # call driver to get function pointers
        _DeviceExp = _ze_device_exp_dditable_t()
        r = ze_result_v(self.__dll.zeGetDeviceExpProcAddrTable(version, byref(_DeviceExp)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.DeviceExp = _DeviceExp

        # attach function interface to function address
        self.zeDeviceGetFabricVertexExp = _zeDeviceGetFabricVertexExp_t(self.__dditable.DeviceExp.pfnGetFabricVertexExp)

        # call driver to get function pointers
        _Context = _ze_context_dditable_t()
        r = ze_result_v(self.__dll.zeGetContextProcAddrTable(version, byref(_Context)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Context = _Context

        # attach function interface to function address
        self.zeContextCreate = _zeContextCreate_t(self.__dditable.Context.pfnCreate)
        self.zeContextDestroy = _zeContextDestroy_t(self.__dditable.Context.pfnDestroy)
        self.zeContextGetStatus = _zeContextGetStatus_t(self.__dditable.Context.pfnGetStatus)
        self.zeContextSystemBarrier = _zeContextSystemBarrier_t(self.__dditable.Context.pfnSystemBarrier)
        self.zeContextMakeMemoryResident = _zeContextMakeMemoryResident_t(self.__dditable.Context.pfnMakeMemoryResident)
        self.zeContextEvictMemory = _zeContextEvictMemory_t(self.__dditable.Context.pfnEvictMemory)
        self.zeContextMakeImageResident = _zeContextMakeImageResident_t(self.__dditable.Context.pfnMakeImageResident)
        self.zeContextEvictImage = _zeContextEvictImage_t(self.__dditable.Context.pfnEvictImage)
        self.zeContextCreateEx = _zeContextCreateEx_t(self.__dditable.Context.pfnCreateEx)

        # call driver to get function pointers
        _CommandQueue = _ze_command_queue_dditable_t()
        r = ze_result_v(self.__dll.zeGetCommandQueueProcAddrTable(version, byref(_CommandQueue)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.CommandQueue = _CommandQueue

        # attach function interface to function address
        self.zeCommandQueueCreate = _zeCommandQueueCreate_t(self.__dditable.CommandQueue.pfnCreate)
        self.zeCommandQueueDestroy = _zeCommandQueueDestroy_t(self.__dditable.CommandQueue.pfnDestroy)
        self.zeCommandQueueExecuteCommandLists = _zeCommandQueueExecuteCommandLists_t(self.__dditable.CommandQueue.pfnExecuteCommandLists)
        self.zeCommandQueueSynchronize = _zeCommandQueueSynchronize_t(self.__dditable.CommandQueue.pfnSynchronize)
        self.zeCommandQueueGetOrdinal = _zeCommandQueueGetOrdinal_t(self.__dditable.CommandQueue.pfnGetOrdinal)
        self.zeCommandQueueGetIndex = _zeCommandQueueGetIndex_t(self.__dditable.CommandQueue.pfnGetIndex)

        # call driver to get function pointers
        _CommandList = _ze_command_list_dditable_t()
        r = ze_result_v(self.__dll.zeGetCommandListProcAddrTable(version, byref(_CommandList)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.CommandList = _CommandList

        # attach function interface to function address
        self.zeCommandListCreate = _zeCommandListCreate_t(self.__dditable.CommandList.pfnCreate)
        self.zeCommandListCreateImmediate = _zeCommandListCreateImmediate_t(self.__dditable.CommandList.pfnCreateImmediate)
        self.zeCommandListDestroy = _zeCommandListDestroy_t(self.__dditable.CommandList.pfnDestroy)
        self.zeCommandListClose = _zeCommandListClose_t(self.__dditable.CommandList.pfnClose)
        self.zeCommandListReset = _zeCommandListReset_t(self.__dditable.CommandList.pfnReset)
        self.zeCommandListAppendWriteGlobalTimestamp = _zeCommandListAppendWriteGlobalTimestamp_t(self.__dditable.CommandList.pfnAppendWriteGlobalTimestamp)
        self.zeCommandListAppendBarrier = _zeCommandListAppendBarrier_t(self.__dditable.CommandList.pfnAppendBarrier)
        self.zeCommandListAppendMemoryRangesBarrier = _zeCommandListAppendMemoryRangesBarrier_t(self.__dditable.CommandList.pfnAppendMemoryRangesBarrier)
        self.zeCommandListAppendMemoryCopy = _zeCommandListAppendMemoryCopy_t(self.__dditable.CommandList.pfnAppendMemoryCopy)
        self.zeCommandListAppendMemoryFill = _zeCommandListAppendMemoryFill_t(self.__dditable.CommandList.pfnAppendMemoryFill)
        self.zeCommandListAppendMemoryCopyRegion = _zeCommandListAppendMemoryCopyRegion_t(self.__dditable.CommandList.pfnAppendMemoryCopyRegion)
        self.zeCommandListAppendMemoryCopyFromContext = _zeCommandListAppendMemoryCopyFromContext_t(self.__dditable.CommandList.pfnAppendMemoryCopyFromContext)
        self.zeCommandListAppendImageCopy = _zeCommandListAppendImageCopy_t(self.__dditable.CommandList.pfnAppendImageCopy)
        self.zeCommandListAppendImageCopyRegion = _zeCommandListAppendImageCopyRegion_t(self.__dditable.CommandList.pfnAppendImageCopyRegion)
        self.zeCommandListAppendImageCopyToMemory = _zeCommandListAppendImageCopyToMemory_t(self.__dditable.CommandList.pfnAppendImageCopyToMemory)
        self.zeCommandListAppendImageCopyFromMemory = _zeCommandListAppendImageCopyFromMemory_t(self.__dditable.CommandList.pfnAppendImageCopyFromMemory)
        self.zeCommandListAppendMemoryPrefetch = _zeCommandListAppendMemoryPrefetch_t(self.__dditable.CommandList.pfnAppendMemoryPrefetch)
        self.zeCommandListAppendMemAdvise = _zeCommandListAppendMemAdvise_t(self.__dditable.CommandList.pfnAppendMemAdvise)
        self.zeCommandListAppendSignalEvent = _zeCommandListAppendSignalEvent_t(self.__dditable.CommandList.pfnAppendSignalEvent)
        self.zeCommandListAppendWaitOnEvents = _zeCommandListAppendWaitOnEvents_t(self.__dditable.CommandList.pfnAppendWaitOnEvents)
        self.zeCommandListAppendEventReset = _zeCommandListAppendEventReset_t(self.__dditable.CommandList.pfnAppendEventReset)
        self.zeCommandListAppendQueryKernelTimestamps = _zeCommandListAppendQueryKernelTimestamps_t(self.__dditable.CommandList.pfnAppendQueryKernelTimestamps)
        self.zeCommandListAppendLaunchKernel = _zeCommandListAppendLaunchKernel_t(self.__dditable.CommandList.pfnAppendLaunchKernel)
        self.zeCommandListAppendLaunchCooperativeKernel = _zeCommandListAppendLaunchCooperativeKernel_t(self.__dditable.CommandList.pfnAppendLaunchCooperativeKernel)
        self.zeCommandListAppendLaunchKernelIndirect = _zeCommandListAppendLaunchKernelIndirect_t(self.__dditable.CommandList.pfnAppendLaunchKernelIndirect)
        self.zeCommandListAppendLaunchMultipleKernelsIndirect = _zeCommandListAppendLaunchMultipleKernelsIndirect_t(self.__dditable.CommandList.pfnAppendLaunchMultipleKernelsIndirect)
        self.zeCommandListAppendImageCopyToMemoryExt = _zeCommandListAppendImageCopyToMemoryExt_t(self.__dditable.CommandList.pfnAppendImageCopyToMemoryExt)
        self.zeCommandListAppendImageCopyFromMemoryExt = _zeCommandListAppendImageCopyFromMemoryExt_t(self.__dditable.CommandList.pfnAppendImageCopyFromMemoryExt)
        self.zeCommandListHostSynchronize = _zeCommandListHostSynchronize_t(self.__dditable.CommandList.pfnHostSynchronize)
        self.zeCommandListGetDeviceHandle = _zeCommandListGetDeviceHandle_t(self.__dditable.CommandList.pfnGetDeviceHandle)
        self.zeCommandListGetContextHandle = _zeCommandListGetContextHandle_t(self.__dditable.CommandList.pfnGetContextHandle)
        self.zeCommandListGetOrdinal = _zeCommandListGetOrdinal_t(self.__dditable.CommandList.pfnGetOrdinal)
        self.zeCommandListImmediateGetIndex = _zeCommandListImmediateGetIndex_t(self.__dditable.CommandList.pfnImmediateGetIndex)
        self.zeCommandListIsImmediate = _zeCommandListIsImmediate_t(self.__dditable.CommandList.pfnIsImmediate)

        # call driver to get function pointers
        _CommandListExp = _ze_command_list_exp_dditable_t()
        r = ze_result_v(self.__dll.zeGetCommandListExpProcAddrTable(version, byref(_CommandListExp)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.CommandListExp = _CommandListExp

        # attach function interface to function address
        self.zeCommandListCreateCloneExp = _zeCommandListCreateCloneExp_t(self.__dditable.CommandListExp.pfnCreateCloneExp)
        self.zeCommandListImmediateAppendCommandListsExp = _zeCommandListImmediateAppendCommandListsExp_t(self.__dditable.CommandListExp.pfnImmediateAppendCommandListsExp)
        self.zeCommandListGetNextCommandIdExp = _zeCommandListGetNextCommandIdExp_t(self.__dditable.CommandListExp.pfnGetNextCommandIdExp)
        self.zeCommandListUpdateMutableCommandsExp = _zeCommandListUpdateMutableCommandsExp_t(self.__dditable.CommandListExp.pfnUpdateMutableCommandsExp)
        self.zeCommandListUpdateMutableCommandSignalEventExp = _zeCommandListUpdateMutableCommandSignalEventExp_t(self.__dditable.CommandListExp.pfnUpdateMutableCommandSignalEventExp)
        self.zeCommandListUpdateMutableCommandWaitEventsExp = _zeCommandListUpdateMutableCommandWaitEventsExp_t(self.__dditable.CommandListExp.pfnUpdateMutableCommandWaitEventsExp)
        self.zeCommandListGetNextCommandIdWithKernelsExp = _zeCommandListGetNextCommandIdWithKernelsExp_t(self.__dditable.CommandListExp.pfnGetNextCommandIdWithKernelsExp)
        self.zeCommandListUpdateMutableCommandKernelsExp = _zeCommandListUpdateMutableCommandKernelsExp_t(self.__dditable.CommandListExp.pfnUpdateMutableCommandKernelsExp)

        # call driver to get function pointers
        _Image = _ze_image_dditable_t()
        r = ze_result_v(self.__dll.zeGetImageProcAddrTable(version, byref(_Image)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Image = _Image

        # attach function interface to function address
        self.zeImageGetProperties = _zeImageGetProperties_t(self.__dditable.Image.pfnGetProperties)
        self.zeImageCreate = _zeImageCreate_t(self.__dditable.Image.pfnCreate)
        self.zeImageDestroy = _zeImageDestroy_t(self.__dditable.Image.pfnDestroy)
        self.zeImageGetAllocPropertiesExt = _zeImageGetAllocPropertiesExt_t(self.__dditable.Image.pfnGetAllocPropertiesExt)
        self.zeImageViewCreateExt = _zeImageViewCreateExt_t(self.__dditable.Image.pfnViewCreateExt)

        # call driver to get function pointers
        _ImageExp = _ze_image_exp_dditable_t()
        r = ze_result_v(self.__dll.zeGetImageExpProcAddrTable(version, byref(_ImageExp)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.ImageExp = _ImageExp

        # attach function interface to function address
        self.zeImageGetMemoryPropertiesExp = _zeImageGetMemoryPropertiesExp_t(self.__dditable.ImageExp.pfnGetMemoryPropertiesExp)
        self.zeImageViewCreateExp = _zeImageViewCreateExp_t(self.__dditable.ImageExp.pfnViewCreateExp)
        self.zeImageGetDeviceOffsetExp = _zeImageGetDeviceOffsetExp_t(self.__dditable.ImageExp.pfnGetDeviceOffsetExp)

        # call driver to get function pointers
        _Mem = _ze_mem_dditable_t()
        r = ze_result_v(self.__dll.zeGetMemProcAddrTable(version, byref(_Mem)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Mem = _Mem

        # attach function interface to function address
        self.zeMemAllocShared = _zeMemAllocShared_t(self.__dditable.Mem.pfnAllocShared)
        self.zeMemAllocDevice = _zeMemAllocDevice_t(self.__dditable.Mem.pfnAllocDevice)
        self.zeMemAllocHost = _zeMemAllocHost_t(self.__dditable.Mem.pfnAllocHost)
        self.zeMemFree = _zeMemFree_t(self.__dditable.Mem.pfnFree)
        self.zeMemGetAllocProperties = _zeMemGetAllocProperties_t(self.__dditable.Mem.pfnGetAllocProperties)
        self.zeMemGetAddressRange = _zeMemGetAddressRange_t(self.__dditable.Mem.pfnGetAddressRange)
        self.zeMemGetIpcHandle = _zeMemGetIpcHandle_t(self.__dditable.Mem.pfnGetIpcHandle)
        self.zeMemOpenIpcHandle = _zeMemOpenIpcHandle_t(self.__dditable.Mem.pfnOpenIpcHandle)
        self.zeMemCloseIpcHandle = _zeMemCloseIpcHandle_t(self.__dditable.Mem.pfnCloseIpcHandle)
        self.zeMemFreeExt = _zeMemFreeExt_t(self.__dditable.Mem.pfnFreeExt)
        self.zeMemPutIpcHandle = _zeMemPutIpcHandle_t(self.__dditable.Mem.pfnPutIpcHandle)
        self.zeMemGetPitchFor2dImage = _zeMemGetPitchFor2dImage_t(self.__dditable.Mem.pfnGetPitchFor2dImage)

        # call driver to get function pointers
        _MemExp = _ze_mem_exp_dditable_t()
        r = ze_result_v(self.__dll.zeGetMemExpProcAddrTable(version, byref(_MemExp)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.MemExp = _MemExp

        # attach function interface to function address
        self.zeMemGetIpcHandleFromFileDescriptorExp = _zeMemGetIpcHandleFromFileDescriptorExp_t(self.__dditable.MemExp.pfnGetIpcHandleFromFileDescriptorExp)
        self.zeMemGetFileDescriptorFromIpcHandleExp = _zeMemGetFileDescriptorFromIpcHandleExp_t(self.__dditable.MemExp.pfnGetFileDescriptorFromIpcHandleExp)
        self.zeMemSetAtomicAccessAttributeExp = _zeMemSetAtomicAccessAttributeExp_t(self.__dditable.MemExp.pfnSetAtomicAccessAttributeExp)
        self.zeMemGetAtomicAccessAttributeExp = _zeMemGetAtomicAccessAttributeExp_t(self.__dditable.MemExp.pfnGetAtomicAccessAttributeExp)

        # call driver to get function pointers
        _Fence = _ze_fence_dditable_t()
        r = ze_result_v(self.__dll.zeGetFenceProcAddrTable(version, byref(_Fence)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Fence = _Fence

        # attach function interface to function address
        self.zeFenceCreate = _zeFenceCreate_t(self.__dditable.Fence.pfnCreate)
        self.zeFenceDestroy = _zeFenceDestroy_t(self.__dditable.Fence.pfnDestroy)
        self.zeFenceHostSynchronize = _zeFenceHostSynchronize_t(self.__dditable.Fence.pfnHostSynchronize)
        self.zeFenceQueryStatus = _zeFenceQueryStatus_t(self.__dditable.Fence.pfnQueryStatus)
        self.zeFenceReset = _zeFenceReset_t(self.__dditable.Fence.pfnReset)

        # call driver to get function pointers
        _EventPool = _ze_event_pool_dditable_t()
        r = ze_result_v(self.__dll.zeGetEventPoolProcAddrTable(version, byref(_EventPool)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.EventPool = _EventPool

        # attach function interface to function address
        self.zeEventPoolCreate = _zeEventPoolCreate_t(self.__dditable.EventPool.pfnCreate)
        self.zeEventPoolDestroy = _zeEventPoolDestroy_t(self.__dditable.EventPool.pfnDestroy)
        self.zeEventPoolGetIpcHandle = _zeEventPoolGetIpcHandle_t(self.__dditable.EventPool.pfnGetIpcHandle)
        self.zeEventPoolOpenIpcHandle = _zeEventPoolOpenIpcHandle_t(self.__dditable.EventPool.pfnOpenIpcHandle)
        self.zeEventPoolCloseIpcHandle = _zeEventPoolCloseIpcHandle_t(self.__dditable.EventPool.pfnCloseIpcHandle)
        self.zeEventPoolPutIpcHandle = _zeEventPoolPutIpcHandle_t(self.__dditable.EventPool.pfnPutIpcHandle)
        self.zeEventPoolGetContextHandle = _zeEventPoolGetContextHandle_t(self.__dditable.EventPool.pfnGetContextHandle)
        self.zeEventPoolGetFlags = _zeEventPoolGetFlags_t(self.__dditable.EventPool.pfnGetFlags)

        # call driver to get function pointers
        _Event = _ze_event_dditable_t()
        r = ze_result_v(self.__dll.zeGetEventProcAddrTable(version, byref(_Event)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Event = _Event

        # attach function interface to function address
        self.zeEventCreate = _zeEventCreate_t(self.__dditable.Event.pfnCreate)
        self.zeEventDestroy = _zeEventDestroy_t(self.__dditable.Event.pfnDestroy)
        self.zeEventHostSignal = _zeEventHostSignal_t(self.__dditable.Event.pfnHostSignal)
        self.zeEventHostSynchronize = _zeEventHostSynchronize_t(self.__dditable.Event.pfnHostSynchronize)
        self.zeEventQueryStatus = _zeEventQueryStatus_t(self.__dditable.Event.pfnQueryStatus)
        self.zeEventHostReset = _zeEventHostReset_t(self.__dditable.Event.pfnHostReset)
        self.zeEventQueryKernelTimestamp = _zeEventQueryKernelTimestamp_t(self.__dditable.Event.pfnQueryKernelTimestamp)
        self.zeEventQueryKernelTimestampsExt = _zeEventQueryKernelTimestampsExt_t(self.__dditable.Event.pfnQueryKernelTimestampsExt)
        self.zeEventGetEventPool = _zeEventGetEventPool_t(self.__dditable.Event.pfnGetEventPool)
        self.zeEventGetSignalScope = _zeEventGetSignalScope_t(self.__dditable.Event.pfnGetSignalScope)
        self.zeEventGetWaitScope = _zeEventGetWaitScope_t(self.__dditable.Event.pfnGetWaitScope)

        # call driver to get function pointers
        _EventExp = _ze_event_exp_dditable_t()
        r = ze_result_v(self.__dll.zeGetEventExpProcAddrTable(version, byref(_EventExp)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.EventExp = _EventExp

        # attach function interface to function address
        self.zeEventQueryTimestampsExp = _zeEventQueryTimestampsExp_t(self.__dditable.EventExp.pfnQueryTimestampsExp)

        # call driver to get function pointers
        _Module = _ze_module_dditable_t()
        r = ze_result_v(self.__dll.zeGetModuleProcAddrTable(version, byref(_Module)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Module = _Module

        # attach function interface to function address
        self.zeModuleCreate = _zeModuleCreate_t(self.__dditable.Module.pfnCreate)
        self.zeModuleDestroy = _zeModuleDestroy_t(self.__dditable.Module.pfnDestroy)
        self.zeModuleDynamicLink = _zeModuleDynamicLink_t(self.__dditable.Module.pfnDynamicLink)
        self.zeModuleGetNativeBinary = _zeModuleGetNativeBinary_t(self.__dditable.Module.pfnGetNativeBinary)
        self.zeModuleGetGlobalPointer = _zeModuleGetGlobalPointer_t(self.__dditable.Module.pfnGetGlobalPointer)
        self.zeModuleGetKernelNames = _zeModuleGetKernelNames_t(self.__dditable.Module.pfnGetKernelNames)
        self.zeModuleGetProperties = _zeModuleGetProperties_t(self.__dditable.Module.pfnGetProperties)
        self.zeModuleGetFunctionPointer = _zeModuleGetFunctionPointer_t(self.__dditable.Module.pfnGetFunctionPointer)
        self.zeModuleInspectLinkageExt = _zeModuleInspectLinkageExt_t(self.__dditable.Module.pfnInspectLinkageExt)

        # call driver to get function pointers
        _ModuleBuildLog = _ze_module_build_log_dditable_t()
        r = ze_result_v(self.__dll.zeGetModuleBuildLogProcAddrTable(version, byref(_ModuleBuildLog)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.ModuleBuildLog = _ModuleBuildLog

        # attach function interface to function address
        self.zeModuleBuildLogDestroy = _zeModuleBuildLogDestroy_t(self.__dditable.ModuleBuildLog.pfnDestroy)
        self.zeModuleBuildLogGetString = _zeModuleBuildLogGetString_t(self.__dditable.ModuleBuildLog.pfnGetString)

        # call driver to get function pointers
        _Kernel = _ze_kernel_dditable_t()
        r = ze_result_v(self.__dll.zeGetKernelProcAddrTable(version, byref(_Kernel)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Kernel = _Kernel

        # attach function interface to function address
        self.zeKernelCreate = _zeKernelCreate_t(self.__dditable.Kernel.pfnCreate)
        self.zeKernelDestroy = _zeKernelDestroy_t(self.__dditable.Kernel.pfnDestroy)
        self.zeKernelSetCacheConfig = _zeKernelSetCacheConfig_t(self.__dditable.Kernel.pfnSetCacheConfig)
        self.zeKernelSetGroupSize = _zeKernelSetGroupSize_t(self.__dditable.Kernel.pfnSetGroupSize)
        self.zeKernelSuggestGroupSize = _zeKernelSuggestGroupSize_t(self.__dditable.Kernel.pfnSuggestGroupSize)
        self.zeKernelSuggestMaxCooperativeGroupCount = _zeKernelSuggestMaxCooperativeGroupCount_t(self.__dditable.Kernel.pfnSuggestMaxCooperativeGroupCount)
        self.zeKernelSetArgumentValue = _zeKernelSetArgumentValue_t(self.__dditable.Kernel.pfnSetArgumentValue)
        self.zeKernelSetIndirectAccess = _zeKernelSetIndirectAccess_t(self.__dditable.Kernel.pfnSetIndirectAccess)
        self.zeKernelGetIndirectAccess = _zeKernelGetIndirectAccess_t(self.__dditable.Kernel.pfnGetIndirectAccess)
        self.zeKernelGetSourceAttributes = _zeKernelGetSourceAttributes_t(self.__dditable.Kernel.pfnGetSourceAttributes)
        self.zeKernelGetProperties = _zeKernelGetProperties_t(self.__dditable.Kernel.pfnGetProperties)
        self.zeKernelGetName = _zeKernelGetName_t(self.__dditable.Kernel.pfnGetName)

        # call driver to get function pointers
        _KernelExp = _ze_kernel_exp_dditable_t()
        r = ze_result_v(self.__dll.zeGetKernelExpProcAddrTable(version, byref(_KernelExp)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.KernelExp = _KernelExp

        # attach function interface to function address
        self.zeKernelSetGlobalOffsetExp = _zeKernelSetGlobalOffsetExp_t(self.__dditable.KernelExp.pfnSetGlobalOffsetExp)
        self.zeKernelSchedulingHintExp = _zeKernelSchedulingHintExp_t(self.__dditable.KernelExp.pfnSchedulingHintExp)
        self.zeKernelGetBinaryExp = _zeKernelGetBinaryExp_t(self.__dditable.KernelExp.pfnGetBinaryExp)

        # call driver to get function pointers
        _Sampler = _ze_sampler_dditable_t()
        r = ze_result_v(self.__dll.zeGetSamplerProcAddrTable(version, byref(_Sampler)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.Sampler = _Sampler

        # attach function interface to function address
        self.zeSamplerCreate = _zeSamplerCreate_t(self.__dditable.Sampler.pfnCreate)
        self.zeSamplerDestroy = _zeSamplerDestroy_t(self.__dditable.Sampler.pfnDestroy)

        # call driver to get function pointers
        _PhysicalMem = _ze_physical_mem_dditable_t()
        r = ze_result_v(self.__dll.zeGetPhysicalMemProcAddrTable(version, byref(_PhysicalMem)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.PhysicalMem = _PhysicalMem

        # attach function interface to function address
        self.zePhysicalMemCreate = _zePhysicalMemCreate_t(self.__dditable.PhysicalMem.pfnCreate)
        self.zePhysicalMemDestroy = _zePhysicalMemDestroy_t(self.__dditable.PhysicalMem.pfnDestroy)

        # call driver to get function pointers
        _VirtualMem = _ze_virtual_mem_dditable_t()
        r = ze_result_v(self.__dll.zeGetVirtualMemProcAddrTable(version, byref(_VirtualMem)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.VirtualMem = _VirtualMem

        # attach function interface to function address
        self.zeVirtualMemReserve = _zeVirtualMemReserve_t(self.__dditable.VirtualMem.pfnReserve)
        self.zeVirtualMemFree = _zeVirtualMemFree_t(self.__dditable.VirtualMem.pfnFree)
        self.zeVirtualMemQueryPageSize = _zeVirtualMemQueryPageSize_t(self.__dditable.VirtualMem.pfnQueryPageSize)
        self.zeVirtualMemMap = _zeVirtualMemMap_t(self.__dditable.VirtualMem.pfnMap)
        self.zeVirtualMemUnmap = _zeVirtualMemUnmap_t(self.__dditable.VirtualMem.pfnUnmap)
        self.zeVirtualMemSetAccessAttribute = _zeVirtualMemSetAccessAttribute_t(self.__dditable.VirtualMem.pfnSetAccessAttribute)
        self.zeVirtualMemGetAccessAttribute = _zeVirtualMemGetAccessAttribute_t(self.__dditable.VirtualMem.pfnGetAccessAttribute)

        # call driver to get function pointers
        _FabricVertexExp = _ze_fabric_vertex_exp_dditable_t()
        r = ze_result_v(self.__dll.zeGetFabricVertexExpProcAddrTable(version, byref(_FabricVertexExp)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.FabricVertexExp = _FabricVertexExp

        # attach function interface to function address
        self.zeFabricVertexGetExp = _zeFabricVertexGetExp_t(self.__dditable.FabricVertexExp.pfnGetExp)
        self.zeFabricVertexGetSubVerticesExp = _zeFabricVertexGetSubVerticesExp_t(self.__dditable.FabricVertexExp.pfnGetSubVerticesExp)
        self.zeFabricVertexGetPropertiesExp = _zeFabricVertexGetPropertiesExp_t(self.__dditable.FabricVertexExp.pfnGetPropertiesExp)
        self.zeFabricVertexGetDeviceExp = _zeFabricVertexGetDeviceExp_t(self.__dditable.FabricVertexExp.pfnGetDeviceExp)

        # call driver to get function pointers
        _FabricEdgeExp = _ze_fabric_edge_exp_dditable_t()
        r = ze_result_v(self.__dll.zeGetFabricEdgeExpProcAddrTable(version, byref(_FabricEdgeExp)))
        if r != ze_result_v.SUCCESS:
            raise Exception(r)
        self.__dditable.FabricEdgeExp = _FabricEdgeExp

        # attach function interface to function address
        self.zeFabricEdgeGetExp = _zeFabricEdgeGetExp_t(self.__dditable.FabricEdgeExp.pfnGetExp)
        self.zeFabricEdgeGetVerticesExp = _zeFabricEdgeGetVerticesExp_t(self.__dditable.FabricEdgeExp.pfnGetVerticesExp)
        self.zeFabricEdgeGetPropertiesExp = _zeFabricEdgeGetPropertiesExp_t(self.__dditable.FabricEdgeExp.pfnGetPropertiesExp)

        # success!
