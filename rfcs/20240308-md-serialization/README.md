# Proposal for memory descriptor serialization

Compilers like Torch Inductor and XLA are gaining popularity and
adding capabilities like Ahead of Time (AOT) compilation.  With those
additions, it puts additional requirements to oneDNN to support
serialization of objects.

In particular, these compiler take as input a computational graph, and
generate an executable binary from it.

For computational graph nodes, these solutions will write to their
output binary the information necessary to recreate a oneDNN primitive
descriptor at runtime.

However, for pre-packed weights, the user cannot generally create the
corresponding memory descriptor as those are generated and queried
from a primitive descriptor. So from oneDNN user perspective, three
possible options exist:
- at compile time, serialize the memory descriptor along with the
  pre-packed weights themselves. At runtime, simply deserialize the
  weights memory descriptor and load weights.
- at compile time, serialize the information necessary to generate a
  primitive descriptor with the weights. At runtime, a primitive
  descriptor is then created and queried for a memory descriptor.


# Proposal: Expose a memory descriptor serialization API.

By exposing a serialization/deserialization API, Inductor and XLA can
serialize pre-packed weights without changing the flow of information
in graph, or the execution flow of these tools.

The main downside here is that it will encourage serializing
pre-packed weights. This is safe only if the serialization and
deserialization envirnoment are exactly the same, since weights layout
can change depending on HW platform, number of threads, ...

AOT Inductor and XLA JIT paths seem ok with that limitation.


## C API

```c
/// Retrieves a binary blob associated with the given memory descriptor
///
/// @param Output blob Pointer to binary blob.
///     If not nullptr, size bytes of the memory descriptor blob are written.
/// @param Output size Pointer to the size of the binary blob in bytes.
///     Size is written if blob is nullptr.
/// @param memory_desc Input memory descriptor to serialize
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_desc_get_blob(
        uint8_t *blob, size_t *size, const_dnnl_memory_desc_t memory_desc);


/// Creates a memory descriptor from a memory descriptor binary blob.
///
/// @param Output memory_desc Pointer to an already allocated buffer.
/// @param blob Pointer to a memory descriptor binary blob
/// @returns #dnnl_success on success and a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_memory_desc_create_with_blob(
        dnnl_memory_desc_t *memory_desc, const uint8_t *blob);


```

This C API will have to be used as follow for serialization and deserialization:
```c++
    // md serialization
    auto wei_desc = pd.weights_desc();
    auto c_wei_desc = wei_desc.get();
    
    size_t size;
    dnnl_memory_desc_get_blob(nullptr, &size, c_wei_desc);
    std::vector<uint8_t> serialized_wei_desc(size);
    dnnl_memory_desc_get_blob(serialized_wei_desc.data(), &size, c_wei_desc);

    // md deserialization
    dnnl_memory_desc_t deserialized_wei_desc;
    dnnl_memory_desc_create_with_blob(
            &deserialized_wei_desc, serialized_wei_desc.data());
```


## C++ API

```c++
        /// Construct a memory descriptor from a binary blob
        ///
        /// @param blob A binary blob previously queried from a memory descriptor.
        memory::desc(const std::vector<uint8_t> &blob);
        
        /// Returns a binary blob associated with the given memory descriptor
        /// @returns The memory descriptor blob associated with the memory descriptor
        std::vector<uint8_t> memory::desc::get_blob();
```
 
And the C++ API usage model:
 
```c++
    // md serialization
    auto wei_desc = pd.weights_desc();
    std::vector<uint8_t> serialized_wei_desc = wei_desc.get_blob();

    // md deserialization
    memory::desc(serialized_wei_desc);
```
 

# Other options considered

Other options were considered from the Inductor and XLA perspectives.
Those were dropped for one or multiple of the following reasons:
a. changes to Inductor JIT compilation that are specific to AOT are
  required. However this goes against the AOT Inductor philosophy
  which is to rely on Inductor JIT compilation as-is.
b. a lot of changes would be required on the framework side to pass
  additional information across graph nodes.
c. complexify the AOT runtime library code and make it
  deviate from PyTorch stock execution flow.


The considered options are:
- serialize plain layout weights and do constant weights caching upon
  first execution. This will hit issue c.
- serialize pre-packed weights, and run an initialization step to fill
  primitive descriptors which contain the weights memory
  descriptors. This would hit issue a. (to generate initialization
  code).
- serialize information to generate a primitive descriptor with
  pre-packed weights. This would hit issue b.
- at runtime, get weights descriptor from the primitive descriptor of
  the computational node. This would hit issues a. and c.
