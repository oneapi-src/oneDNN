# Add Persistent Cache API for OpenCL Engine

## Motivation
In order to get all required information about GPU device oneDNN has to
compile a dummy OpenCL kernel to use the compiler binary for querying that
information.
The kernel compilation happens during engine creation. In order to mitigate
the associated overhead oneDNN caches the compiled binaries however, this
doesn't help when the engine is created for the first time.

OpenVINO reported that first engine creation overhead makes their network
loading mechanism twice as slow.

There is an option to rely on CL cache (implemented in driver). The main
drawback of the option is that it requires to modify registry on Windows which
doesn't work for some OpenVINO customers. Because of that OpenVINO has its own
caching mechanism that caches serialized OpenCL kernels on disk.

## Proposal

oneDNN already provides persistent cache API for OpenCL runtime that allowed
OpenVINO to eliminate first primitive creation overhead. OpenVINO can use the
same approach to eliminate first engine creation.

The proposal is to introduce persistent cache API for engine. The API will be
OpenCL runtime specific therefore it will reside in OpenCL interop namespace.

### C API
```c
// dnnl_ocl.h

/// Retrieves a cache blob ID for the OpenCL device.
///
/// @warning
///     This API is intended to be used with
///     #dnnl_ocl_interop_engine_get_cache_blob() and
///     #dnnl_ocl_interop_engine_create_from_cache_blob(). The returned cache
///     blob ID can only be used as an ID of the cache blob returned by
///     #dnnl_ocl_interop_engine_get_cache_blob().
///
/// @note The cache blob ID can be empty (@p size will be 0 and
///     @p cache_blob_id will be nullptr) if oneDNN doesn't have anything to
///     put in the cache blob. (#dnnl_ocl_interop_engine_get_cache_blob will
///     return an empty cache blob).
///
/// @param device An OpenCL device.
/// @param size Size of the cache blob ID in bytes.
/// @param cache_blob_id Cache blob id of size @p size. If
///     the @p cache_blob_id is nullptr then the size of the cache blob ID is
///     returned in @p size.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t dnnl_ocl_interop_engine_get_cache_blob_id(
        cl_device_id device, size_t *size, uint8_t *cache_blob_id);

/// Retrieves a cache blob associated with the given engine.
///
/// @note The cache blob can be empty (@p size will be 0 and @p cache_blob
///     will be nullptr) if oneDNN doesn't have anything to put in the cache
///     blob. It's the user's responsibility to check whether it's empty
///     prior to passing it to
///     #dnnl_ocl_interop_engine_create_from_cache_blob().
///
/// @param engine Engine to query for the cache blob.
/// @param size Size of the cache blob in bytes.
/// @param cache_blob Cache blob of size @p size. If the @p cache_blob is
///     nullptr then the size of the cache blob is returned in @p size.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t dnnl_ocl_interop_engine_get_cache_blob(
        dnnl_engine_t engine, size_t *size, uint8_t *cache_blob);

/// Creates an engine from the given cache blob.
///
/// @param engine Output engine.
/// @param device The OpenCL device that this engine will encapsulate.
/// @param context The OpenCL context (containing the device) that this
///     engine will use for all operations.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
/// @param size Size of the cache blob in bytes.
/// @param cache_blob Cache blob of size @p size.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t dnnl_ocl_interop_engine_create_from_cache_blob(
        dnnl_engine_t *engine, cl_device_id device, cl_context context,
        size_t size, const uint8_t *cache_blob);
```
### C++ API
```cpp
// dnnl_ocl.hpp

namespace ocl_interop {

/// Returns the cache blob ID of the OpenCL device.
///
/// @warning
///     This API is intended to be used with
///     #dnnl::ocl_interop::get_engine_cache_blob() and
///     #dnnl::ocl_interop::make_engine(device, context, cache_blob).
///     The returned cache blob ID can only be used as an ID of the cache blob
///     returned by #dnnl::ocl_interop::get_engine_cache_blob().
///
/// @note The cache blob ID can be empty (@p size will be 0 and
///     @p cache_blob_id will be nullptr) if oneDNN doesn't have anything to
///     put in the cache blob. (#dnnl_ocl_interop_engine_get_cache_blob will
///     return an empty cache blob).
///
/// @param device An OpenCL device.
/// @returns A vector containing the cache blob ID.
inline std::vector<uint8_t> get_engine_cache_blob_id(cl_device_id device);

/// Returns a cache blob for the engine.
///
/// @note The cache blob vector can be empty if oneDNN doesn't have anything
///     to put in the cache blob. It's the user's responsibility to check
///     whether it's empty prior to passing it to
///     #dnnl::ocl_interop::make_engine(device, context, cache_blob).
///
/// @param engine Engine to query for the cache blob.
/// @returns Vector containing the cache blob.
inline std::vector<uint8_t> get_engine_cache_blob(const engine &aengine);

/// Constructs an engine from the given cache blob.
///
/// @param device The OpenCL device that this engine will encapsulate.
/// @param context The OpenCL context (containing the device) that this
///     engine will use for all operations.
/// @param cache_blob Cache blob.
/// @returns An engine.
inline engine make_engine(cl_device_id device, cl_context context,
        const std::vector<uint8_t> &cache_blob);

} // ocl_interop
```

## Limitations
Since OpenCL doesn't provide a sustainable way to get a stepping of a device
there is a limitation for the proposed persistent cache API.
The limitation is that the cached blobs cannot be re-used for devices that
have different stepping because oneDNN cannot differentiate such devices.

## Implementation Details
Content of cache blob and cache blob ID is implementation detail and is not
specified therefore users cannot make any assumption on it.

This section contains information about the implementation details of the
cache blob and cache blob ID.

### Cache Blob ID
The cache blob ID will contain the following information:
1. Driver version
2. oneDNN version
3. Device name
4. Platform name
5. oneDNN hash

### Cache Blob

The cache blob will contain information about the given device that was queried
from the compiled binary and not that compiled binary itself for the following
reasons:
1. The compiled binary is not stored anywhere. In order to put the binary in
a cache blob upon calling the corresponding API it should be stored in engine
2. Information about binary kind (OpenCL or Level Zero) must be stored along
with the binary to let the nGEN know how to handle it
3. Size of the queried information is 118 bytes at the moment of writing this
RFC, while size of the binary is 1432 (Gen 9, OpenCL)
4. Deserialization is faster as it doesn't involve querying information from
the compiled binary
