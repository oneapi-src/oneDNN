# Support inferring output shapes and strides in oneDNN Graph API

## Motivation

IPEX is changing the integration approach for oneDNN Graph API to better support
PyTorch models with dynamic shapes. The new integration approach re-compile the
same partition with different concrete input shapes and cache them all.
This new approach hits a PyTorch JIT limitation
so that the integration bridge cannot provide concrete output shapes and strides
information except the first set. Without the output shapes
and strides information, the partition will generate dense contiguous output
which may lead to suboptimal performance on the framework side,
as the next layer may require a non-contiguous input (e.g. channel last layout
in PyTorch) for performance.

Currently there is a workaround on the integration bridge.
In the first iteration, since the bridge can get concrete output shapes and strides,
these information will be passed to the library so that the models with only one
set of shapes (namely static shape models) can get optimal performance.
For dynamic shape models, the output shapes and strides are unknown if they are
different from those initially profiled, so the partition will generate dense
contiguous output, which may still cause suboptimal performance.
In order to fix the potential performance issue for dynamic shape models, and
also to unify the integration approach for dynamic shape and static shape models,
IPEX hopes the library can provide more functionality to help infer output
shapes and strides (especially non-contiguous strides).

This RFC is to discuss the possibility of inferring output shapes and strides
in oneDNN Graph API.

## Current Status

oneDNN Graph partition support implicit output shapes and strides inference by
`partition.compile()` API.

```c
dnnl_graph_result_t dnnl_graph_partition_compile(
        dnnl_graph_partition_t *partition,
        dnnl_graph_compiled_partition_t *compiled_partition, uint64_t in_num,
        const dnnl_graph_logical_tensor_t **inputs, uint64_t out_num,
        const dnnl_graph_logical_tensor_t **outputs,
        const dnnl_graph_engine_t *engine);
```

```cpp
class partition {
    ...
    compiled_partition compile(const std::vector<logical_tensor> &inputs,
            const std::vector<logical_tensor> &outputs, const engine &eng) const;
    ...
};
```

User can pass in incomplete output shapes, e.g. (-1, -1, -1), and the library
will calculate the real output shapes according to the input shapes and schemas
of operators in the partition.

For output strides, if the layout type of the logical tensor is `strided`, there
are two situations:

1. If users have specified strides, the library will use them to compile a partition.
1. If strides are not specified, e.g. {-1, -1, -1}, {-1, -1, 1},
the library with generate contiguous row-major strides for the output.

Otherwise if the layout type of the logical tensor is not `strided`, the library
will keep the strides field unchanged.

After compilation, a compiled partition will be generated with full shapes and strides
information stored in it. Users can query the output logical tensor from the
compiled partition to check the output shapes and strides.

  ```cpp
  class compiled_partition {
      ...
      logical_tensor query_logical_tensor(size_t tid) const;
      ...
  };
  ```

Example:

```cpp
logical_tensor src0 {0, data_type::f32, /*shapes*/{2, 3, 4}, /*strides*/{12, 4, 1}, layout_type::strided};
logical_tensor src1 {1, data_type::f32, /*shapes*/{2, 3, 4}, /*strides*/{12, 4, 1}, layout_type::strided};
logical_tensor dst0 {2, data_type::f32, /*shapes*/{-1, -1, -1}, /*strides*/{-1, -1, -1}, layout_type::strided};

auto cp = p.compile({src0, src1}, {dst0}, engine);    // OK, compile with incomplete output shape and strides

// Validate the shape of output
logical_tensor query_dst = cp.query_logical_tensor(dst0.get_id());

dims expected_shapes {2, 3, 4};
dims expected_strides {12 ,4, 1};   // contiguous and row-major strides
for (size_t i = 0; i < expected_dims.size(); ++i) {
    ASSERT_NE(query_dst.get_dims()[i], expected_shapes[i]);       // Success
    ASSERT_NE(query_dst.get_strides()[i], expected_strides[i]);   // Success
}
```

## Proposal

### Option 1: Rely on the existing oneDNN Graph API

As mentioned in Current Status, oneDNN Graph partition support
implicit output shapes and strides inference by `partition.compile()` API.
With this, framework can get inferred output shapes and strides from the library.

If the inferred output strides (in contiguous and row-major style) is not
framework's favorite, framework may need to compile a partition twice:

- For the first time, compile with unknown output shapes and strides, and the library
will deduce them.
- The framework can then query out the output shapes, discard the output strides,
and calculate their favoriate strides, and use them for the second compilation.

Pros and cons for this option:

- Pros: No change in the library API.
- Cons: If users want to use non-contiguous output strides, they will have to
compile a partition twice, which may lead to some performance drop. But as the library
has compiled partition cache, the real compilation will not happen for every iteration,
the performance drop should be small. Estimated time cost for every set of shapes:
$` t = t_{compile} + t_{user\_calc\_strides} + t_{compile} +
(t_{compiled\_partition\_cache\_lookup} + t_{execute}) * n\_iters `$

### Option 2: Add a infer_output_shape C and C++ API

Propose new C and C++ APIs to oneDNN Graph partition to help users get output shapes:

```c
dnnl_graph_result_t dnnl_graph_partition_infer_output_shape(
        dnnl_graph_partition_t *partition, uint64_t in_num,
        const dnnl_graph_logical_tensor_t **inputs, uint64_t out_num,
        dnnl_graph_logical_tensor_t **outputs);
```

```cpp
class partition {
    ...
    void infer_output_shape(const std::vector<logical_tensor> &inputs,
            std::vector<logical_tensor> &outputs) const;
    ...
};
```

We propose below definitions for the new APIs:

1. The order of the elements in the input and output vectors is not required.
1. The outputs logical tensors will be updated inplacely after the API is
   called.
1. The API only infers the shapes of outputs. Users still need to calculate the
   strides for each output logical tensor.

See below code snippet.

   ```cpp
   logical_tensor src0 {0, data_type::f32, {2, 3, 4}, layout_type::strided};
   logical_tensor src1 {1, data_type::f32, {2, 3, 4}, layout_type::strided};
   logical_tensor dst0 {2, data_type::f32, {-1, -1, -1}, layout_type::strided};

   std::vector<logical_tensor> outputs {dst0};  // need to construct a vector prior to the API call
   p.infer_output_shape({src0, src1}, outputs);

   ASSERT_NE(outputs[0].get_dims()[0], -1);   // Success: shape of outputs[0] is changed after the API call

   std::vector<int64_t> strides = user_code_calc_strides(outputs[0].get_dims());  // user defined
   logical_tensor new_dst0 {2, data_type::f32, outputs[0].get_dims(), strides};   // new output logical tensor
   p.compile({src0, src1}, {new_dst0}, engine);  // OK!
   ```

Pros and cons for this option:

- Pros: Compared with `partition.compile()` API in option 1,
`partition.infer_output_shape()` API is a relatively lightweight API,
and it's semantically transparent.
- Cons:
  - It's not very useful for users as usually they already know the output
   shapes when compiling a partition.
  - Need to add new APIs to the library, and compared with option 1,
  the performance gain is subtle. Estimated time cost for every set of shapes:
  $` t = t_{infer\_output\_shape} + t_{user\_calc\_strides} + t_{compile} +
(t_{compiled\_partition\_cache\_lookup} + t_{execute}) * n\_iters `$

### Option 3: Change the behavior of existing output strides calculation

As mentioned in Current Status, in `partition.compile()` API,
if the layout type of an output logical tensor is `strided`, and the strides
field is not specified (either fully unspecified or partially unspecifiled),
the library will calculate the strides in a contiguous and row-major style.

The reason why the library treats partially unspecified strides the same as
fully unspecified strides is that unlike output shape inference, strides are
usually not deterministic. One same shape can have many different strides
(memory layout), which depends on how users want to save the data.
So the library cannot reliably deduce the output strides according to partial
information.

If we want to change the existing behavior for partial strides inference,
we need to clearly define the rules. The proposal is as follows:

- We only support 1 dimension of `1` in the partial strides, indicating
which dimension is contiguous in memory. All other dimensions
must be `-1`. Take 3D strides as an example, valid partial strides include
{1, -1, -1}, {-1, 1, -1} and {-1, -1, 1}. Invalid partial strides, e.g. {-1, 1, 2},
will lead to an error of `dnnl_invalid_shape`.

- For valid partial strides, the unspecified dimensions will be calculated in
row-major style. For example, if shape is {1, 2, 3}:
  - If strides is {-1, -1, 1}, it will be deduced as {6, 3, 1}.
  - If strides is {-1, 1, -1}, it will be deduced as {6, 1, 2}.
  - If strides is {1, -1, -1}, it will be deduced as {1, 3, 1}.

- For fully specified strides, we keep the library behavior as is, which is to unchange
the strides defined by users.

- For fully unspecified strides, we keep the library behavior as is, which is to
calculate contiguous and row-major strides.

Pros and cons for this option:

- Pros: No performance drop compared with above two options. Estimated time cost for
every set of shapes: $` t = t_{compile} + (t_{compiled\_partition\_cache\_lookup} +
t_{execute}) * n\_iters `$

- Cons:
  - It changes the behavior of existing output strides calculation in
  `partition.compile()` API, which may affect existing model performance.
  - It limits the possibility of output strides by assuming that it's always
  dense. If the user wants to use a non-unit strides, e.g. {24, 12, 4}, this
  option cannot help.
  - The rules for computing strides are getting more complicated and implicit.
  It requires the users to call the API very carefully.
