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

The API will be responsible for providing a user with an ID for the binary
representation of a kernel and the binary representation itself.

* ID is an array of bytes represented as `std::vector<unsigned char>` that is a
unique identifier for a binary representation. ID should take into account that
binaries compiled for different devices are different binaries. The binaries
compiled for a sub-device considered compatible with the main device.
* Binary representation is an array of bytes represented as `std::vector<unsigned char>`.
Since oneDNN primitive can consist of multiple kernels the API return
`std::vector<std::vector<unsigned char>>`.

This proposal covers only providing binary representations of kernels. However,
it's possible that oneDNN will support serialization of the primitives in the future
so that users can cache the whole model on disk. Given that, it makes sense to
make API more common and scalable.

The suggestion is to introduce a binary kind to distinguish binary representation of
a kernel from primitive.

```cpp
struct primitive {
    enum class binary_kind {
        kernel,
        /*primitive - unimplemented */
    };
    /* ... */
};
```

The API for obtaining ID will be provided by the `primitive_desc_base` class
therefore the API will be available in each `primitive_desc` class.

```cpp
struct primitive_desc_base {
    /* ... */
    std::vector<unsigned char> get_id() const;
};
```

The `primitive` class will get a new constructor to take a binary and new API
to provide user with the binary.

```cpp
struct primitive {
    /* ... */
    primitive(const primitive_desc &pd, const std::vector<std::vector<unsigned char>> &binaries, binary_kind kind);
    // Primitives that are not supported return an empty vector.
    std::vector<std::vector<unsigned char>> get_binaries(binary_kind kind) const;
};
```

##### API usage example

```cpp
using namespace dnnl;

{
    convolution_forward::primitive_desc conv_pd(desc, attr, engine);
    primitive conv(conv_pd);

    auto key = conv_pd.get_id();
    auto binaries = conv.get_binaries(primitive::binary_kind::kernel);
    // Store the binaries on disk.
    store_to_persistent_cache(key, binaries);
}

{
    convolution_forward::primitive_desc conv_pd(desc, attr, engine);
    auto key = conv_pd.get_id();
    // Load binaries from disk.
    auto binaries = load_from_persistent_cache(key);
    primitive conv_from_cache(pd, binary, primitive::binary_kind::kernel);
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
