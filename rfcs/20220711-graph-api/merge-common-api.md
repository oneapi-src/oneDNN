# Common APIs for oneDNN Graph

We expose runtime APIs of engine, stream, and threadpool in oneDNN Graph API.
These APIs share the similar semantics with oneDNN's engine, stream, and
threadpool, respectively. Previously, in oneDNN Graph technical previews, we
designed these runtime APIs separately rather than re-using them from oneDNN to
reduce the dependency on oneDNN implementation. In oneDNN v3.0, the APIs and
implementations of Graph API will be moved into oneDNN source tree, it will make
more sense to merge and unify these runtime APIs and implementation.

In this document, we will discuss the design of merging runtime APIs and sharing
them in both primitive and graph API. We will start from engine API which is the
basic for the other two.

Please note graph APIs mentioned in this document are based on [oneDNN graph
v0.5 release](https://github.com/oneapi-src/oneDNN/releases/tag/graph-v0.5)
which is slightly different from what they look on the latest dev-graph.

## Existing engine API

Taking the C++ API as an example, oneDNN's engine contains below public APIs:

```cpp
/// in dnnl.hpp
struct engine {

    /// Kinds of engines.
    enum class kind {
        /// An unspecified engine
        any = dnnl_any_engine,
        /// CPU engine
        cpu = dnnl_cpu,
        /// GPU engine
        gpu = dnnl_gpu,
    };

    /// Constructs an empty engine. An empty engine cannot be used in any
    /// operations.
    engine() = default;

    /// Returns the number of engines of a certain kind.
    ///
    /// @param akind The kind of engines to count.
    /// @returns The number of engines of the specified kind.
    static size_t get_count(kind akind);

    /// Constructs an engine.
    ///
    /// @param akind The kind of engine to construct.
    /// @param index The index of the engine. Must be less than the value
    ///     returned by #get_count() for this particular kind of engine.
    engine(kind akind, size_t index);

    /// Constructs an engine based on a primitive from the primitive
    /// descriptor @p pd by querying its engine.
    ///
    /// @param pd The primitive descriptor to query.
    engine(const handle<dnnl_primitive_desc_t> &pd);

    /// Returns the kind of the engine.
    /// @returns The kind of the engine.
    kind get_kind() const;

    /// Returns the engine of a primitive descriptor.
    ///
    /// @param pd The primitive descriptor to query.
    /// @returns A weak handle to the engine that the primitive descriptor was
    ///     created with.
    template <typename primitive_desc>
    static engine query(const primitive_desc &pd);
};
```

Besides, oneDNN also provides `make_engine()`, `get_context()`, and
`get_device()` APIs as SYCL and OpenCL inter-operation API.

The existing engine defined in oneDNN Graph API:

```cpp
/// in dnnl_graph.hpp
class engine {
public:
    /// engine kind
    enum class kind {
        /// An unspecified engine
        any = dnnl_graph_any_engine,
        /// CPU engine
        cpu = dnnl_graph_cpu,
        /// GPU engine
        gpu = dnnl_graph_gpu,
    };

    /// Constructs an engine with specified kind and device_id
    ///
    /// @param akind The kind of engine to construct
    /// @param device_id Specify which device to be used
    engine(kind akind, int device_id);

    /// Constructs an engine with specified kind and device_id
    ///
    /// @param akind Engine kind
    /// @param device_id Specify which device to be used
    /// @param alloc The memory allocator bound with engine
    engine(kind akind, int device_id, allocator &alloc);

    /// Set allocator to an engine
    ///
    /// @param alloc The memory allocator bound with engine
    void set_allocator(allocator &alloc);

    /// Returns device handle of the current engine
    ///
    /// @returns Device handle
    void *get_device_handle();

    /// Returns device id of the current engine
    ///
    /// @returns Device id
    int get_device_id() const;

    /// Returns concrete kind of the current engine
    ///
    ///@returns Kind of engine
    kind get_kind() const;
};
```

oneDNN Graph API technical preview also provided `make_engine()` as SYCL
inter-operation API.

1. We have the same engine kind enumeration in both primitive and graph API and
   propose to keep them in oneDNN v3.0.
2. We propose to keep the default constructor `engine()` and constructor
   `engine(kind, int)` in oneDNN v3.0.
3. We propose to keep the API `get_kind()` in oneDNN v3.0.
4. We propose to keep the API `make_engine()`, `get_device()`, and
   `get_context()` in SYCL inter-operation API. Graph API will not support
   OpenCL inter-operation API.
5. We propose to remove the API `get_device_handle()` and `get_device_id()` from
   Graph API because they are not used in the framework integration and can be
   covered by `get_device()` and `get_context()` mentioned above in SYCL
   inter-operation API.
6. The main difference between primitive and graph API is allocator. It will be
   discussed separately the following section.
7. Besides, we consider that `set_allocator()` member function is redundant as
   we can ask users to always provide allocator at engine constructor and keep
   it constant during engine lifetime. Allowing users to set another allocator
   is less useful but increasing the maintenance effort of the API.

## Engine implementation

The oneDNN engine base class contains the following members:

- engine_kind_
- runtime_kind_
- index_
- counter_

For the derived SYCL engine, it also contains:

- sycl device
- sycl context
- backend_ (is a enum class for all available sycl backends, including host, level0, opencl, nvidia)

The oneDNN Graph engine contains the following members:

- device_handle_
- kind_
- device_id_
- allocator_
- cl::sycl::device
- cl::sycl::context

From these, most members in oneDNN Graph engine can be mapped to the members in
oneDNN engine, except the `device_handle_` and `allocator_`. The
`device_handle_` is actually not used for now, so we propose to remove it from
the engine implementation in oneDNN v3.0. Then the only gap will be the
allocator in oneDNN Graph API which will be discussed below.

## Allocator API

We will have 4 options to handle the allocator API and its implementation in
oneDNN v3.0.

### Option 1: keep separate engine and allocator API in Graph API

It's possible for us to keep the engine and allocator API in Graph API as is.
The option will takes the least effort as we will not need to change any API or
implementation for both primitive and graph. But as a single product, having two
engine concepts in API will confuse users and make the API redundant. In the
backend of graph component, we still need to convert the "graph" engine to
"primitive" engine which will introduce unnecessary overhead. Besides, if we
don't merge "graph" engine and "primitive" engine together, we will not be able
to merge the two streams either.

### Option 2: decouple engine and allocator in Graph API

We can decouple engine and allocator in Graph API by moving allocator out of
engine and its implementation. With this, we can merge graph engine and
primitive engine easily without changing the API and implementation and keep the
allocator as a separate concept of Graph API only.

User code example before the change:

```cpp
using namespace dnnl::graph;

// construct engine with allocator
allocator alloc {f_alloc, f_delete};
engine eng {engine::kind::cpu, 0, alloc};

// compile a partition
part.compile(inputs, outputs, eng);
```

After the change:

```cpp
using namespace dnnl;

// construct allocator and engine separately
graph::allocator alloc {f_alloc, f_delete};
engine eng {engine::kind::cpu, 0};

// compile a partition
part.compile(inputs, outputs, eng, alloc);
```

1. This option requires big refactor to Graph API and implementation. Basically,
   the option requires users to pass an allocator and an engine at the places
   where a graph engine is required at this moment. C API changes as follows:

    ```cpp
    // Current Graph API
    dnnl_graph_result_t dnnl_graph_tensor_create(
            dnnl_graph_tensor_t **tensor,
            const dnnl_graph_logical_tensor_t *logical_tensor,
            const dnnl_graph_engine_t *engine, void *handle);
            
    dnnl_graph_result_t dnnl_graph_partition_compile(
            dnnl_graph_partition_t *partition,
            dnnl_graph_compiled_partition_t *compiled_partition, uint64_t in_num,
            const dnnl_graph_logical_tensor_t **inputs, uint64_t out_num,
            const dnnl_graph_logical_tensor_t **outputs,
            const dnnl_graph_engine_t *engine);
            
    // We need to change them to
    dnnl_graph_result_t dnnl_graph_tensor_create(
            dnnl_graph_tensor_t **tensor,
            const dnnl_graph_logical_tensor_t *logical_tensor,
            const dnnl_engine_t *engine, const dnnl_graph_allocator_t *allocator, void *handle);

    dnnl_graph_result_t dnnl_graph_partition_compile(
            dnnl_graph_partition_t *partition,
            dnnl_graph_compiled_partition_t *compiled_partition, uint64_t in_num,
            const dnnl_graph_logical_tensor_t **inputs, uint64_t out_num,
            const dnnl_graph_logical_tensor_t **outputs,
            const dnnl_engine_t *engine, const dnnl_graph_allocator_t *allocator);
    ```

2. Besides the big change scope, another downside of the option is in C++ API,
   we will mix use components from dnnl namespace and dnnl::graph namespace
   together in one API like below. It can be mitigated by having an alias of
   engine in dnnl::graph namespace.

    ```cpp
    // in dnnl::graph:: namespace
    class partition {
        // ...

        // engine is from dnnl:: while allocator and logical tensor are from dnnl::graph::.
        void compile(std::vector<logical_tensor> &inputs, std::vector<logical_tensor> &outputs, const dnnl::engine &eng, const allocator &alloc);

        // ...
    };
    ```

3. The option also implies a flexible combination of engine and allocator which
   means an engine can be used along with several different allocator objects.
   This was not allowed previously in Graph API technical previews and we don't
   know if it has real use cases.

Both option 1 and option 2 will not change the implementation of oneDNN engine
in oneDNN v3.0. But the following option 3 and option 4 will require changes to
the engine implementation.

### Option 3: make an engine with graph inter-interoperability

With this option, we will not change the engine API in oneDNN but the
implementation of oneDNN engine to hold an opaque `void *` allocator, just like
how SYCL device is holden in SYCL engine. Then we can add an inter-operability
API to create a "graph" engine.

The graph inter-operability API:

```cpp
// C API, in dnnl_graph.h.
dnnl_status_t dnnl_graph_make_engine_with_allocator(dnnl_engine_t* engine, dnnl_engine_kind_t kind, int index, const dnnl_graph_allocator_t* allocator);

// C++ API, in namespace dnnl::graph::.
dnnl::engine make_engine(dnnl::engine::kind akind, size_t index, const allocator &alloc);
```

### Option 4: make the allocator part of primitive API

If we consider the allocator API to be also useful to primitives, we can make it
part of primitive API by moving the allocator related APIs from
`dnnl_graph.h/hpp` to `dnnl.h/hpp`. The users can create both allocator and
engine by directly using primitive API. Because the primitives will not respect
a user-provided allocator in the current implementation, we need to document it
clearly and add corresponding checks in the implementation.

The engine C++ API changes:

```cpp
// in dnnl.hpp

struct engine {
    // ...

    // a new constructor with allocator
    engine(kind akind, size_t index, const allocator &alloc);

    // a new setter
    void set_allocator(const allocator &alloc);

    // ...
};

```

This option will make the primitive API self-contained and clean, but it also
requires big changes to primitive API, implementation, and documents. It's also
confusing if allocator is supported in API but not used in primitive
implementation in a short time.

### Option 5: decouple engine and allocator and maintain a singleton inside the library

The option was proposed on the review meeting. It suggests to decouple engine
and allocator on Graph API (similar as option 2) and provide a global setter for
users to specify allocators to the library. Internally, the library will store
the allocators and use them where they are needed.

```cpp
// construct allocator and engine separately
dnnl::graph::allocator alloc {f_alloc, f_delete};
dnnl::engine eng {dnnl::engine::kind::cpu, 0};

// new API: set allocator to the library
dnnl_graph_set_allocator(dnnl::engine::kind::cpu, alloc);

// compile a partition, use the alloc internally
part.compile(inputs, outputs, eng);
```

With this option, compared with option 2, there will be no need to change Graph
API everywhere and be able to keep oneDNN engine intact. We just need to add a
new API to specify allocator and use the allocator along with oneDNN engine.
Internally, we need to change graph implementation to save allocators and
provide global singleton access.

Because CPU memory allocation and SYCL memory allocator have different function
signature, users may need to set different allocator for different engine kind.
And internally, the library need to keep different singletons for host and
devices and dispatch the allocation and de-allocation according to engine kind.

With this option, calling `dnnl_graph_set_allocator` multiple times for the same
engine kind will be ignored. Only the allocator set at the first time will be
saved and used by the library. Also, if `dnnl_graph_set_allocator` is never
called, default allocators should be provided and used internally.

A summary for the above options:

| options | primitive API change | primitive impl change | graph API change | graph impl change | clean API | low overhead |
|---------|----------------------|-----------------------|------------------|-------------------|-----------|--------------|
| option1 | No                   | No                    | No               | No                | No        | No           |
| option2 | No                   | No                    | Big              | Big               | Yes       | Yes          |
| option3 | No                   | Small                 | Small            | No                | Yes       | Yes          |
| option4 | Big                  | Moderate              | No               | No                | Yes       | Yes          |
| option4 | No                   | No                    | Moderate         | Moderate          | Yes       | Yes          |

The recommendation goes to option 4. With this option, we can fully reuse the
`stream` and `thread pool` APIs in graph API.

## Graph component implementation

With merging graph engine into primitive engine, we can always use the primitive
engine in graph component implementation.

- In Graph C API, we will use type `dnnl_engine_t` from primitive types. It
  means we need to include `dnnl_types.h` in `dnnl_graph_types.h`.

  ```cpp
  // in dnnl_graph.h
  dnnl_graph_result_t dnnl_graph_partition_compile(
      dnnl_graph_partition_t *partition,
      dnnl_graph_compiled_partition_t *compiled_partition, uint64_t in_num,
      const dnnl_graph_logical_tensor_t **inputs, uint64_t out_num,
      const dnnl_graph_logical_tensor_t **outputs,
      const dnnl_engine_t *engine);
  ```

- In graph C++ API, we will use the class `dnnl::engine` from primitive C++ API.
  It means we need to include `dnnl.hpp` in `dnnl_graph.hpp`.

  ```cpp
  // in dnnl_graph.hpp, namespace dnnl::graph::.
  class partition {
      // ...

      void compile(std::vector<logical_tensor> &inputs,
          std::vector<logical_tensor> &outputs,
          const dnnl::engine &eng);

      // ...
  };
  ```

- In graph component implementation, we will need to include the headers
  `c_types_map.hpp` and `engine.hpp` from `src/common` and call
  `dnnl::impl::engine_t` from them.
- In graph's DNNL backend, before calling primitive C++ API, we need to
  construct the `dnnl::engine` object from `dnnl::impl::engine_t`.

  ```cpp
  dnnl::engine make_dnnl_engine(const dnnl::impl::engine_t &e) {
      dnnl::engine eng;
      eng.reset(const_cast<dnnl::impl::engine_t *>(&e), true);
      return eng;
  }
  ```
