# Persistent Cache in oneDNN

## Motivation

oneDNN provides only in-memory primitive cache capabilities which is not
sufficient for the scenarios when time of the first model run is significant,
for example in the case of GPU primitives. An average GPU primitive creation time
is 400ms comparing to less than a millisecond for CPU primitives. The original
request came from OpenVINO but other customers are also affected.

## Proposal

For addressing overhead of the first primitive creation a persistent cache can
be used. Persistent cache is aimed at resolving such issues by storing compiled
kernels on disk.

### Option 1: Implement Persistent Cache Inside oneDNN

The persistent cache is fully implemented in oneDNN. It is represented as a
directory that is specified by a user. The directory contains compiled kernels
that can be loaded to reduce time of the first creation.

#### API

oneDNN provides API to specify directory for the primitive cache and for specifying
persistent cache capacity in megabytes.

```cpp
// Persistent cache can be enabled/disabled for a particular engine kind.
// At this point, we can postpone this API and simply implement persistent cache
// only for GPU to avoid unnecessary overhead for CPU primitives.
inline status set_persistent_cache(int enable, engine::kind kind);
inline bool get_persistent_cache(engine::kind kind);

// Default capacity is 0 - persistent cache is disabled.
inline status set_persistent_cache_capacity(int capacity);
inline int get_persistent_cache_capacity();

// Default dir is working directory.
inline status set_persistent_cache_dir(const std::string &dir);
```

#### Key and Value

Both, the key and value are represented as array of bytes. All values that affect
equality of kernel binary representations should be serialized to become the key,
this also includes version of the driver and git hash of oneDNN library.
The existing key for primitive cache can be partially (no need in all data members)
serialized for creating the aforementioned array of bytes.

The value is simply a binary representation of a kernel for GPU or CPU.

#### Persistent Cache Structure

As it has been already mentioned the persistent cache is basically a directory
where the compiled kernels are stored.

* The 1st level subdirectories represent buckets. The name of the buckets is a hash
value for the key, which the bucket is associated with
* The 2nd level subdirectories (inside buckets) represent persistent cache entries.
The first entry named 0, the second named 1 and so on.
* Inside the 2nd level directories there are two files:
    * key.bin - the binary representation of the key to handle collisions
    * kernel.bin - binary representation of the compiled kernel
* At the top level there is a file named `size.txt` that contains information about
current memory consumption
* (Optionally for DPCPP runtime) There may be an optional 1st level directory
to distinguish between SYCL backends - OpenCL and Level Zero. While Level Zero
supports OpenCL binary format, OpenCL doesn't support Level Zero binary format
(`zebin`) yet.

Suppose a user specified `~/persistent_cache` via `set_persistent_cache_dir`,
then the directory layout will be as follows:

```
~/persistent_cache
├── size.txt
├── 14037140681559929700
│   ├── 0
│   │   ├── key.bin
│   │   └── kernel.bin
│   └── 1
│       ├── key.bin
│       └── kernel.bin
└── 12125041384284752820
    └── ...
        └── ...
```

#### Memory Consumption

* Option 1: Implement FIFO eviction policy that will be relying on `creation time`
of the `kernel.bin` file. The oldest one will be removed. The reason why LRU policy
cannot be implemented is because `last access` information may not be properly
updated depending on different OSes and file systems
* Option 2: User's application is responsible for clearing persistent cache to
avoid excessive consuming disk space. oneDNN doesn't provide any API to clear
persistent cache. The API for setting capacity can be used to specify the limit
for memory consumption. If the capacity is less than the size specified in
`size.txt` oneDNN doesn't add new kernels to the persistent cache.

Proposal is to go with the option 1 (FIFO eviction policy). Since oneDNN is
responsible for storing data on disk it should also be responsible for managing
memory consumption. This will only require applications to set needed capacity
and oneDNN will take care of the rest.

#### Load

When the requested primitive was not found in the primitive cache the persistent
cache is used to find the binary representation of the kernel. If found, the
primitive is created using the binary representation otherwise primitive creation
is performed from scratch.

#### Store

Option 1: Binary representation is stored every time when the requested primitive
is created from scratch
Option 2: Binary representations that are in the primitive cache at the time of
primitive cache destruction are stored in the persistent cache

The proposal is to go with the option 1 to avoid unnecessary dependency between
persistent cache and primitive cache. This is expected that primitive cache miss
doesn't occur often in workloads otherwise the performance will be affected either
way.

#### Interprocess and Thread Synchronization

The access to the persistent cache should be performed in a process and thread
safe manner. For performance reasons both process and thread synchronization
should be performed using readers-writer lock.

### Option 2: Provide API for Implementing Persistent Cache

oneDNN provides an API so that users can implement persistent cache in their
applications.

#### API

The primitive descriptor and primitive abstractions will get a new API:
* Primitive
    * API to provide a cache blob associated with the primitive
    * API to create the primitive from the cache blob
* Primitive descriptor
    * API to provide a cache blob ID that serves as a unique identifier for the
      cache blob

The cache blob ID and cache blob objects are represented as one-dimensional
`uint8_t` arrays.

Note: The format and content of cache blob ID and cache blob arrays are not
specified.

Note: Git hash will affect equality of the cache blob IDs.

The capability the API provides is different from serialization because the
cache blob object cannot be used alone to reconstruct the associated primitive.
Users will have to create and provide the corresponding primitive descriptor
along with the cache blob to create the primitive.

The API for obtaining the cache blob ID and cache blob arrays is defined in the
next sections.

##### Querying Cache Blob ID

The primitive descriptor query mechanism will be extended to support querying
the size of the cache blob ID and a pointer to the cache blob ID. Similar to the
other queries, the pointer to cache blob ID is valid only during the lifetime
of the primitive descriptor.

###### C API
```c
// dnnl_types.h

/// Query kind                      | Type of query result
/// --------------------------------|-----------------------------
/// ...                             | ...
/// dnnl_query_cache_blob_id        | const uint8_t **

typedef enum {
    dnnl_query_undef = 0, ///< no query
    // ... //

    // New queries
    dnnl_query_cache_blob_id_size_s64, ///< size of cache blob ID in bytes
    dnnl_query_cache_blob_id, ///< cache blob ID (pointer to array)

    // memory and op descriptor section
    dnnl_query_some_d = 64, ///< stub
    // ... //
} dnnl_query_t;
```

###### C++ API
```cpp
struct primitive_desc_base : public handle<dnnl_primitive_desc_t> {
// ... //
    std::vector<uint8_t> get_cache_blob_id() const;
};
```

##### Querying Cache Blob

The API for querying the cache blob from a primitive is different from other
query API because it doesn't return a pointer to encapsulated cache blob but
writes the cache blob content to the passed buffer. The rationale behind that
is that the cache blob is not stored inside the primitive and is constructed on
demand during the query call. 

###### C API
```c
/// Retrieves a cache blob associated with the given primitive.
///
/// @param primitive Primitive to query for the cache blob.
/// @param size Size of the cache blob in bytes.
/// @param cache_blob Cache blob of size @p size. If the @p cache_blob is
///     nullptr then the size of the cache blob is returned in @p size.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_get_cache_blob(
        const_dnnl_primitive_t primitive, size_t *size, uint8_t *cache_blob);

/// Creates a primitive from a cache blob.
///
/// @param primitive Output primitive.
/// @param primitive_desc Primitive descriptor used to create the primitive.
/// @param size Size of the cache blob in bytes.
/// @param blob Cache blob of size @p size.
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_primitive_create_from_cache_blob(
        dnnl_primitive_t *primitive, const_dnnl_primitive_desc_t primitive_desc,
        size_t size, const uint8_t *cache_blob);
```

###### C++ API
A new constructor that takes a blob will be added to primitive classes for each
primitive kind.
There is an alternative to extend all existing primitive constructors so that they
take an optional argument `blob`. I think that having the distinct constructors
will keep the API clean and intuitive.

```cpp
struct primitive : public handle<dnnl_primitive_t> {
    /// Returns a cache blob for the primitive.
    ///
    /// @returns Vector containing the cache blob.
    std::vector<uint8_t> get_cache blob() const;
};

struct convolution_forward : public primitive {
    /// Constructs a <primitive name> primitive.
    /// @param pd Primitive descriptor for a convolution forward propagation
            primitive.
    /// @param cache_blob Vector that containing the cache blob.
    convolution_forward(const primitive_desc &pd,
            const std::vector<uint8_t> &cache_blob) : primitive(pd) {}
};
```

##### API Usage Example
```cpp
using namespace dnnl;
{
    convolution_forward::primitive_desc conv_pd(desc, attr, engine);
    convolution_forward conv(conv_pd);
    std::vector<uint8_t> key = conv_pd.get_cache_blob_id();
    std::vector<uint8_t> value = conv.get_cache_blob();
    store_cache_blob_on_disk(key, value);
}

{
    convolution_forward::primitive_desc conv_pd(desc, attr, engine);
    std::vector<uint8_t> key = conv_pd.get_cache_blob_id();
    std::vector<uint8_t> value = load_cache_blob_from_disk(key);
    convolution_forward conv_from_cache_blob(conv_pd, value);
}
```

## Options Comparison

The options are very different and have their pros and cons.

### Option 1

Pros:
* Users can use the feature out-of-the-box

Cons:
* Not a flexible solution
* Implementation and maintenance complexity
* Not all users may want to have thread or process safe cache, which has certain
overhead
* Eviction policy is fixed. Users may want to have their own eviction policy
* Users can implement store/load more efficiently, e.g. store/load batch of binaries (packed),
while oneDNN can only store/load binaries per each primitive creation
* Doesn't scale for primitive serialization

### Option 2

Pros:
* Flexible solution, users can implement:
    * Eviction policy
    * Memory consumption management
    * Interprocess/thread communication
    * Load/unload mechanism
    * Tracing mechanism
* Provides a minimum to implement persistent cache according to the user's needs
* Scalable for primitive serialization
* Applications that use oneDNN can implement any API that is more suitable for their
users

Cons:
* Users have to implement persistent cache on their own

### Conclusion
The proposal is to go with the option 2 to provide a minimum functionality so
that users can implement a persistent cache that fulfils their needs.

## Scope of Support
* Support only binary representation of a kernel
* Support only GPU
    * Start with primitives that are used by OpenVINO
    * Extend the list of supported primitives by request
* No support for CPU primitive at this point

