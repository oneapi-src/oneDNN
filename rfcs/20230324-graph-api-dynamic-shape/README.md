# Proposal for adding dynamic shape support in Graph API

## 1. Background

Dynamic shape is getting more and more attention in deep learning model enabling
and performance optimization. In contrast to static shape, dynamic shape means
that the input or output shape of a model or a specific layer is changing from
iteration to iteration. The typical usages of dynamic shape include:

- The variable sequence length in nature language processing models (e.g.,
  BERT).
- The variable mini-batch size in real-time inference serving.
- The variable spatial size of object detection models in computer vision.

Currently in oneDNN, primitive kernels are usually generated in time (a.k.a.
JIT) according to src/dst shapes, data types, and other information provided by
users through primitive creation API. In some cases, kernel generation is time
consuming, especially on GPU. In order to reduce the kernel generation time,
oneDNN provides a primitive cache mechanism to cache the kernels and reuse them
if the same problem is encountered next time. But oneDNN will have to generate
the kernel again if a different problem shape is provided by users. There are
two main drawbacks for the primitive cache design:

- It does not help for dynamic shape scenario if the shape changes constantly in
  a wide range during the execution time. In this case, the primitive cache will
  probably miss and kernel re-generation will be triggered repeatedly.
- Even for the static shape scenario, primitive cache does not help for the
  cases where first-time inference latency is critical for users. For this case,
  a persistent cache mechanism is exposed to mitigate the issue.

oneDNN provides limited dynamic shape support for few primitives (e.g., matmul
and reorder). It's achieved by supporting [runtime
dimensions](https://oneapi-src.github.io/oneDNN/dev_guide_matmul.html#general-notes)
at primitive creation stage. As mentioned, currently, runtime dimension is not
supported by all primitives. Even for matmul primitive, once runtime dimension
is specified, post ops are not fully supported due to [implementation
limitations](https://github.com/oneapi-src/oneDNN/blob/master/src/common/primitive_attr.cpp#L223).

Similar problems and design also exist in oneDNN Graph API. On partition
compilation stage, concrete problem shapes should be provided by users directly
or can be inferred from the input shapes. A compiled partition cache mechanism
is provided to cache the generated kernels for a specific problem. When shape
changes, re-compilation should be triggered by users. It means that for dynamic
shape scenario, users will not be able to compile a partition once ahead of time
and use it repeatedly for the following iterations.

## 2. Motivation

The main purposes of this proposal are to:

- Explore the possibility of compiling partitions with dynamic shape to reduce
  the overhead of re-compilation.
- Support the usage of "compile once, use in every iteration" from the
  perspective of frameworks.
- Extend the library operation set and programming model to support shape
  tensors and shape manipulation operations which can enlarge the partition
  subgraph and deliver more optimization opportunity to the library backends.
- Expose the runtime dimension feature of primitive API through graph API.

## 3. API proposals

### 3.1 Graph building stage

At the graph building stage, API users create a graph and add operations
associated with input and output logical tensors to the graph. The users also
need to specify the ID, data type, and dimension information for the input and
output logical tensors.

#### 3.1.1 Dimension representation

With the existing Graph API, input and output shapes may be unknown at graph
building stage but should be determined at partition compilation stage.
Currently, INT64_MIN
([DNNL_GRAPH_UNKNOWN_DIM](https://oneapi-src.github.io/oneDNN/group_dnnl_graph_api_logical_tensor.html#macros))
is used to represent an unknown dimension, 0 is used to represent 0-size
dimension, and positive values are used to represent explicit dimension sizes.
For example, shape (INT64_MIN, INT64_MIN, 32, 32) means the shape has 4
dimensions (`ndims = 4` in the definition of logical tensor) with the first two
dimensions unknown. Shapes containing unknown dimensions can be used to define
input and output logical tensors at graph building stage or the output tensors
at partition compilation stage. For the latter case, the output shape should be
deducible according to the shape inference rules of each operation.

To extend the support to dynamic shape problem, this rfc proposes to define
another wildcard value `DNNL_GRAPH_DYNAMIC_DIM` as the placeholder for dynamic
dimensions. It brings two main benefits to differentiate "unknown dimensions"
and "dynamic dimensions":

1. It helps to keep the backward compatibility for INT64_MIN dimension as
  unknown dimension.
2. It allows to check dynamic dimension and skip the related ops graph
   partitioning if a backend does not support dynamic dimension.

| API                        | Value              | Description                                                   |
|:---------------------------|:-------------------|:--------------------------------------------------------------|
| DNNL_GRAPH_UNKNOWN_DIM     | INT64_MIN          | Unknown dimension but can be determined at compilation stage. |
| **DNNL_GRAPH_DYNAMIC_DIM** | INT64_MIN + 1      | Dynamic dimension which is unknown until execution stage.     |

With that, users need to set the unknown or dynamic dimensions carefully at
graph building stage and partition compilation stage. Currently it’s proposed to
only support dense contiguous layout for inputs and outputs with dynamic
dimension. If an input shape contains dynamic dimension, the strides (if user
provided) should be filled with `DNNL_GRAPH_DYNAMIC_DIM` in all dimensions or
directly set layout type to `strided`. The library implementation will check the
validity of strides if explicitly provided.

Calling API `get_mem_size()` on a logical tensor containing dynamic dimensions
will return `(size_t)DNNL_GRAPH_DYNAMIC_DIM`. It follows the similar
behavior of memory descriptor in primitive API. When a memory descriptor
has dimension of `DNNL_RUNTIME_DIM_VAL`, calling `dnnl::memory::desc::get_size()`
will return
[`DNNL_RUNTIME_SIZE_VAL`](https://oneapi-src.github.io/oneDNN/group_dnnl_api_memory.html#macros).

<details>
    <summary>Static shape example code</summary>

```cpp
/// unknown ndims is allowed
logical_tensor in0 {0, data_type::f32, {UNKNOWN_DIM, UNKNOWN_DIM, UNKNOWN_DIM}, layout_type::undef};
logical_tensor in1 {1, data_type::f32, {UNKNOWN_DIM, UNKNOWN_DIM, UNKNOWN_DIM}, layout_type::undef};
logical_tensor out {2, data_type::f32, {UNKNOWN_DIM, UNKNOWN_DIM, UNKNOWN_DIM}, layout_type::undef};
/// create op
op matmul {0, op::kind::MatMul, {in0, in1}, {out}, "matmul_op"};
/// create graph with a specific engine kind
graph g(engine::kind::cpu);
/// add op to graph
g.add_op(matmul);
```

</details>

<details>
    <summary>Dynamic shape example code</summary>

```cpp
/// the first dimension is marked as dynamic，ndims must be known
logical_tensor in0 {0, data_type::f32, {DYNAMIC_DIM, UNKNOWN_DIM, UNKNOWN_DIM}, layout_type::undef};
logical_tensor in1 {1, data_type::f32, {DYNAMIC_DIM, UNKNOWN_DIM, UNKNOWN_DIM}, layout_type::undef};
logical_tensor out {2, data_type::f32, {DYNAMIC_DIM, UNKNOWN_DIM, UNKNOWN_DIM}, layout_type::undef};
/// create op
op matmul {0, op::kind::MatMul, {in0, in1}, {out}, "matmul_op"};
/// create graph with a specific engine kind
graph g(engine::kind::cpu);
/// add op to graph
g.add_op(matmul);
```
</details>

#### 3.1.2 Shape tensor and shape manipulation operations

As aforementioned, some specific layers can also contain dynamic dimensions even
though the input of the whole model is static. For example, an input tensor can
be reshaped or transposed into another shape and the target shape is computed
online as part of the model execution. To support these cases, we propose to
introduce `DynamicReshape` and `DynamicTranspose` operations. The two operations
will accept a `src` tensor as its first input argument and a target shape tensor
as its second input argument. In some models, the target shape information is
calculated by the previous layer and stored in the shape tensor during
execution.

Detailed definition for the new operations can be found at
[DynamicReshape](./DynamicReshape.md) and
[DynamicTranspose](./DynamicTranspose.md).

### 3.2 Graph partitioning stage

During partitioning stage, the library walks through the given graph and
backends decide the partitions according to their fusion capabilities. Users get
returned partitions and rewrite the original framework graph.

If a dimension will be dynamic at compilation stage, when building the graph,
users should represent it with `DNNL_GRAPH_DYNAMIC_DIM`. If a backend cannot
support compilation for dynamic dimension for a certain partition, it should
detect dynamic dimensions early at partitioning stage and should not claim the
partition.

### 3.3 Partition compilation stage

For compilation with static shape inputs, users are required to provide concrete
shape information for all input logical tensors. As of output logical tensors,
the shapes can be unknown as long as they can be inferred according to inputs.
After compilation, a compiled partition is returned as an executable object,
which contains generated codes specialized for the input shapes, layout,
datatype, etc.

For dynamic shape problems, users need to represent dynamic dimensions with
`DNNL_GRAPH_DYNAMIC_DIM` and then pass them to the compilation API. Providing
determinable dimensions (INT64_MIN or any other positive values) at graph
building stage but dynamic dimensions (`DNNL_GRAPH_DYNAMIC_DIM`) at compilation
stage will lead to a partition compilation error. Below table shows two examples
of this kind of incompatibility.

| Graph building          | Partition compilation  |
|:------------------------|:-----------------------|
| [1, 1, 1, 1]            | [1, 1, DYNAMIC_DIM, 1] |
| [1, DYNAMIC_DIM, 1, 1]  | [1, 1, 1, 1]           |

### 3.4 Execution stage

Creating a tensor which binds to a logical tensor containing dynamic dimensions
will raise error.

#### 3.4.1 Shape consistency

Once compiled, the returned compiled partition can be executed multiple times
with different shapes. However, the shapes provided at execution stage should
be also consistent and compatible with compilation stage.

| Case No.    | Compilation            | Execution              |
|:------------|:-----------------------|:-----------------------|
| 1           | [1, 1, DYNAMIC_DIM, 1] | [1, 1, 1, 1]           |
| 2           | [1, 1, DYNAMIC_DIM, 1] | [1, 1, 10, 1]          |
| 3           | [1, 1, DYNAMIC_DIM, 1] | [1, 10, 10, 1]         |
| 4           | [1, 1, DYNAMIC_DIM, 1] | [1, 1, DYNAMIC_DIM, 1] |

Case 1 and 2 in the table will work as expected. For case 3, the second
dimension is a positive (or determinable) value `1` at compilation while at
execution time a different positive value `10` is provided which is inconsistent
with provided previously via compile API. For case 4, a concrete shape must be
provided at execution time.

#### 3.4.2 Infer output shapes

In the existing programming model, after partition compilation, users need to
query the output logical tensors from the compiled partition, check the required
memory sizes, and allocate memory buffer for each output tensor. Unlike static
shape problem, for dynamic shape, users will not provide concrete dimension
during compilation, hence the compiled partition does not contain full shape and
size information for the output tensors.

Given that users are still responsible for allocating output buffers when
executing a compiled partition, this rfc proposes to add a new API to help infer
and return all output logical tensors according to given input logical tensors
(associated with concrete dimensions) from a compiled partition which is
compiled with dynamic shapes.

Proposed C++ API is as below:

```cpp
class compiled_partition {
public:
    // ...
  
    /// Returns inferred output logical tensors. This API is dedicated for
    /// dynamic shape case, users need provide input logical tensors with
    /// concrete input shapes and then library can help infer output shape.
    /// Other than dynamic dimensions, those determinable dimensions (denoted as
    /// INT64_MIN or any positive value) should be the same as the values which
    /// are passed to compilation API before, otherwise an exception will be
    /// raised.
    ///
    /// @param inputs The input logical tensors with concrete shapes.
    /// @returns A list of output logical tensors.
    std::vector<logical_tensor> infer_outputs(
            const std::vector<logical_tensor> &inputs) const;
    
    // ...
}
```

Accordingly new C API will be added.

```c
/// Returns the number of output logical tensors from a compiled partition
///
/// @param compiled_partition The handle of target compiled_partition.
/// @param num Number of output logical tensors
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_compiled_partition_get_outputs_num(
        const_dnnl_graph_compiled_partition_t compiled_partition, size_t *num);

/// Returns inferred output logical tensors. This API is dedicated for
/// dynamic shape case, users need provide input logical tensors with concrete
/// input shapes and then library can help infer output shape. Other than
/// dynamic dimensions, those determinable dimensions (denoted as INT64_MIN or
/// any positive value) should be the same as the values which are passed to
/// compilation API before.
///
/// @param compiled_partition The handle of target compiled_partition.
/// @param num_outputs Number of output logical tensors to be queried
/// @param outputs The output logical tensors to be queried.
/// @param num_inputs Number of input logical tensors
/// @param inputs The input logical tensors
/// @returns #dnnl_success on success or a status describing the error
///     otherwise.
dnnl_status_t DNNL_API dnnl_graph_compiled_partition_infer_outputs(
        const_dnnl_graph_compiled_partition_t compiled_partition,
        size_t num_outputs, dnnl_graph_logical_tensor_t *outputs,
        size_t num_inputs, const dnnl_graph_logical_tensor_t **inputs);
```

<details>
    <summary>Example code to demonstrate the usage</summary>

```cpp
/// ----------------- 1. Graph building ------------------------
/// create input and output logical tensors
/// the first dimension is marked as dynamic
logical_tensor in0 {0, data_type::f32, {DYNAMIC_DIM, UNKNOWN_DIM, UNKNOWN_DIM}, layout_type::undef};
logical_tensor in1 {1, data_type::f32, {DYNAMIC_DIM, UNKNOWN_DIM, UNKNOWN_DIM}, layout_type::undef};
logical_tensor out {2, data_type::f32, DNNL_GRAPH_UNKNOWN_NDIMS, layout_type::undef};
/// create op
op matmul {0, op::kind::MatMul, {in0, in1}, {out}, "matmul_op"};
/// create graph with a specific engine kind
graph g(engine::kind::cpu);
/// add op to graph
g.add_op(matmul); g.finalize();

/// get partitions from the graph
auto partitions = g.get_partitions();

/// ----------------- 2. Compilation ------------------------
/// prepare input logical tensors
/// the first dimension is marked as dynamic (should be consistent with graph building stage)
logical_tensor in0_new {0, data_type::f32, {DYNAMIC_DIM, 3, 5}, layout_type::strided};
logical_tensor in1_new {1, data_type::f32, {DYNAMIC_DIM, 5, 9}, layout_type::strided};
logical_tensor out_new {2, data_type::f32, DNNL_GRAPH_UNKNOWN_NDIMS, layout_type::strided};

/// create engine
engine e(engine::kind::cpu, 0);
/// compile a partition
compiled_partition cp = partitions[0].compile({in0_new, in1_new}, {out_new}, e);

/// ----------------- 3. Execution ------------------------
/// at execution stage, dynamic dimensions should be determinable and known
logical_tensor in0_tmp {0, data_type::f32, {4, 3, 5}, layout_type::strided};
logical_tensor in1_tmp {1, data_type::f32, {4, 5, 9}, layout_type::strided};
std::vector<logical_tensor> outputs_with_shape = cp.infer_outputs({in0_tmp, in1_tmp}); // <------ new API

/// Now output logical tensor has complete shape information, and users can
/// find the corresponding logical tensor and allocate output buffer based on it.

/// prepare input and output tensors
tensor in0_ts {in0_tmp, e, in_buffer0};
tensor in1_ts {in1_tmp, e, in_buffer1};
tensor out_ts {outputs_with_shape[0], e, out_buffer};
/// create stream
stream strm {e};
/// execute a compiled partition
```
</details>

## 4. Implementation Considerations

### 4.1 New backend API

`infer_outputs()` - implementation of corresponding frontend API.

Each backend should implement this interface by itself.

```cpp
    /// Infer output logical tensors according to input logical tensors
    /// @param out_lts The inferred output logical tensors
    /// @param in_lts A list of input logical tensors with concrete shapes
    /// @return The status code.
    virtual status_t infer_outputs(
            const std::vector<logical_tensor_t *> &out_lts,
            const std::vector<const logical_tensor_t *> &in_lts) const = 0;
```

### 4.2 Backend support

Each backend should clarify the supporting capability of dynamic shape when
defining a fusion pattern.

The initial proposal is to fully rely on the graph compiler backend to handle
and optimize partitions with dynamic dimensions. The DNNL backend should be
enhanced to detect dynamic dimension at partitioning stage and claim partitions
accordingly.

In the future, the DNNL backend will work with primitive API to define the
capability, requirements to primitive API, and expose the dynamic shape support
from the DNNL backend.

EOD
