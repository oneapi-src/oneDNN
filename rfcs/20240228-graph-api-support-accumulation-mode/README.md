# Graph API: Add Accumulation Mode Support in oneDNN Graph

## Introduction

Please refer to the [dcoumentation](https://oneapi-src.github.io/oneDNN/dev_guide_attributes_accumulation_mode.html#doxid-dev-guide-attributes-accumulation-mode) 
and [rfc](https://github.com/mgouicem/oneDNN/blob/mgouicem/rfcs/f16-accumulation/rfcs/20230118-f16-accumulation/README.md) for the motivation and design detail of primitive API.

Currently, oneDNN uses f32 for floating point computation and s32 for integer 
computation as the default accumulation data types. However, on some platforms, 
using smaller accumulation data types can result in additional speed 
improvements.

By introducing support for the accumulation data type, oneDNN can achieve up to 
a 2x speedup while maintaining a high level of accuracy for f16 inference. It is
important to note that f16 accumulation is not suitable for training, which
requires full f32 precision.

This document proposes the corresponding accumulation mode API for oneDNN Graph.
To provide control granularity and ease of use, the API offers four options:
global setting, graph-level attribute, partition-level attribute, and op-level
attribute.

## Accumulation mode definition

oneDNN graph will reuse the C and C++ API for accumulation mode enumerations
defined by primitive API:

C API:

```c
typedef enum {
    dnnl_accumulation_mode_strict,  
    dnnl_accumulation_mode_relaxed, 
    dnnl_accumulation_mode_any,     
    dnnl_accumulation_mode_s32,     
    dnnl_accumulation_mode_f32,     
    dnnl_accumulation_mode_f16,     
} dnnl_accumulation_mode_t;
```

C++ API:

```cpp
enum class accumulation_mode {
    strict = dnnl_accumulation_mode_strict,
    relaxed = dnnl_accumulation_mode_relaxed,
    any = dnnl_accumulation_mode_any,
    s32 = dnnl_accumulation_mode_s32,
    f32 = dnnl_accumulation_mode_f32,
    f16 = dnnl_accumulation_mode_f16,
}
```

The accumulation mode attribute accepts:
- `strict` (default): For floating-point datatypes, the default accumulation
datatype is `f32`(or f64 for f64 primitives). For integral datatypes, the 
default accumulation datatype is `s32`.
- `relaxed`: Same as strict except some partial accumulators can be
  rounded to the src/dst datatype in memory.
- `any`: Uses the fastest implementation available with one of the
  src/dst datatypes or a higher precision accumulation datatype.
- `f32`, `f16` and `s32`: Uses the specified accumulation datatype.

 Framework users may need map their own accumulation mode definitions to the
 above enumerations.

## Proposal

### Option 1: Specify accumulation mode thorugh global function

Use a global function to set the accumulation mode, which will affect all the
created graphs. Users should set the accumulation mode at the graph build stage
before compilation.

The C API will be like:

```c
status_t DNNL_GRAPH_API
dnnl_graph_set_accumulation_mode(dnnl_graph_accumulation_mode_t mode);

dnnl_graph_accumulation_mode_t DNNL_GRAPH_API 
dnnl_graph_get_accumulation_mode();
```

The correspodning C++ API will be like:

```cpp
namespace dnnl {
namespace graph {
void set_accumulation_mode(accumulation_mode mode);
}
}
```

#### Pros and Cons

Pros:

- No change on all existing APIs

Cons:

- Only provide coarse-grained control on all graphs
- Need efforts for managing global status.
- Require users to provide essential information at the early graph build stage 

### Option 2: Support accumulation mode on graph level( Recommended )

All the computations or operations in a graph will be specified with the same
accumulation data type. Whether the given accumulation mode can be utilized is
decided by the backend capbility. 

This option refers to the current solution for floating-point match mode
attribute support in oneDNN Graph. These attributes share some similarities to
a certain extent: both provide a certain degree of speedup by modifying the
datatype based on backend capabilities.

The API will be like:

```cpp
// for demonstrate purpose, we may provide the graph parameter in a wrapper or
// define more constructors.
graph g(kind, accumulation_mode);
op foo(id, kind, "foo"); g.add_op(foo);
op bar(id, kind, "bar"); g.add_op(bar);
partitions p = g.get_partitions();

// compile and generate kernel according to the accumulation mode setting on
// graph all the partitions from the same graph should share the same math mode
compiled_partition cp0 = p[0].compile(inputs, outputs, engine);
compiled_partition cp1 = p[1].compile(inputs, outputs, engine);
cp0.execute(…);
cp1.execute(…);
```

Since we already have the `fpmath_mode` attribute for graph, oneDNN Graph should
support for combination for these two attributes.

#### Pros and Cons

Pros:

1. No changes are needed for op-related and partition-related operations, such
as op creation and compilation.
2. Graph may provide different fusion strategies based on the given accumulation
mode according to backend capabilities.
3. Aligned with the existing similar math mode attribute, it might be easier for
users to handle them together.

Cons:

1. Relatively coarse-grained control, no support for partition-level and
op-level attribute setting. 
2. All ops in the graph will be set the same accumulation mode, yet in practice
it's possible that only a subset of the subgraphs will use f16 accumulation
( e.g. the first layer in the NN might not use this attribute ). Users should
handle such case by themselves such as creating multiple graphs.
3. Require users to provide essential information at the early graph build stage
4. For multi-thread compilation, cannot specify accumulation mode for each
thread.

### Option 3: Support accumulation mode on partition level

Users will be asked to provide the accumulation mode during compilation stage.
In this case, users may need to set the accumulation mode for different
partitions separately.

The API will be like:

```cpp
graph g(kind);
op foo(id, kind, "foo"); g.add_op(foo);
op bar(id, kind, "bar"); g.add_op(bar);
partitions p = g.get_partitions();
// compile and generate kernel according to the accumulation mode setting
// here different partition may be compiled with different accumulation
// mode
compiled_partition cp0 = p[0].compile(inputs, outputs, engine, accumulation_mode0);
cp0.execute(…);

compiled_partition cp1 = p[1].compile(inputs, outputs, engine, accumulation_mode1);
cp1.execute(…);
```

#### Pros and Cons

Pros:

1. Not require users to provide the accumulation mode information until
compilation stage.
2. Users may specify different accumulation mode for different backends.
3. Users may compile the same partition with different accumulation mode.
4. Scalable for multi-threading cases.

Cons:

1. If a subset of ops in the subgraph are not required to use f16 accumulation
mode( e.g. imagine only the first matmul in the sdp fusion uses f16
accumulation), users need to handle such case by themselves.
2. The library need to ensure the partition is acceptable on the given
accumulation mode

### Option 4: Support accumulation mode on op level

Users will be asked to set the accumulation mode as an attribute when creating
an op. The API will be like:

```cpp
graph g(kind);

std::string acc_mode1{"strict"};
std::string acc_mode2{"relax"};

// set accumulation_mode as an attribute
op foo(id, kind, "foo"); foo.set_attr<std::string>(op_attr::accumulation_mode, acc_mode1);
op bar(id, kind, "bar"); bar.set_attr<std::string>(op_attr::accumulation_mode, acc_mode2);
g.add_op(foo);
g.add_op(bar);
// partitioning algorithm needs to respect accumulation mode on the
// operators. may not fuse if two operators have different accumulation 
// modes
partitions p = g.get_partitions();

// compile and generate kernel according to the accumulation mode
// setting on op. different partition may have different accumulation mode
compiled_partition cp0 = p[0].compile(inputs, outputs, engine);
cp0.execute(…);

compiled_partition cp1 = p[1].compile(inputs, outputs, engine);
cp1.execute(…);
```

#### Pros and Cons

Pros:

1. Most fine-grained control over the operations.
2. If part of the ops in the subgraph are not required to set f16 accumulation
mode, the library will be able to do pattern matching accordingly.

Cons:

1. Huge code change on operation API and integration.
2. Might introduce more complex logic for fusion.

## Verbose Support

The specified accumulation mode will be printed in the verbose in the manner
like `accm:f16`.

## Validation

Currently benchdnn has already supported setting accumulation mode for
validation, such as:

```cpp
--attr-acc-mode=MODE
```

benchdnn graph should be able to specify the accumulation mode in the JSON file
based on the implementation. For instance, if the accumulation mode is added as
a graph attribute, it can be validated like:

```JSON
{
  "version": "3.0.0",
  "engine_kind": "cpu",
  "accumulation_mode": "strict",
  "graph": [ 
    //... 
  ]
}
```

If the accumulation mode is enabled through a global API or partition-level
API, benchdnn graph needs to be updated to accommodate it. Additionally, the
validation input should be similar with the requirements of the graph-level
API. This will enable benchdnn graph to parse the JSON file and configure the
accumulation mode accordingly.

If the accumulation mode is added as an operation attribute, it might be
validated like:

```JSON
{
      "id": 0,
      "name": "Convolution",
      "kind": "Convolution",
      "attrs": {
        "strides": {
          "type": "s64[]",
          "value": [
            2,
            2
          ]
        },
        "accumulation_mode": {
          "type": "string",
          "value": "strict"
        },
        //...
      }
      //...
}
```

END
