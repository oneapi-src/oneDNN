# Graph API: Add Accumulation Mode Support in oneDNN Graph

## Introduction

Please refer to the [documentation](https://oneapi-src.github.io/oneDNN/dev_guide_attributes_accumulation_mode.html#doxid-dev-guide-attributes-accumulation-mode) 
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

### Option 1: Specify accumulation mode through global function

Use a global function to set the accumulation mode, which will affect all the
created graphs. Users will be required to set this attribute before compilation
stage.

The C API will be like:

```c
status_t DNNL_GRAPH_API
dnnl_graph_set_accumulation_mode(dnnl_accumulation_mode_t mode);

dnnl_accumulation_mode_t DNNL_GRAPH_API 
dnnl_graph_get_accumulation_mode();
```

The corresponding C++ API will be like:

```cpp
namespace dnnl {
namespace graph {
void set_accumulation_mode(accumulation_mode mode);
}
}
```

#### Pros and Cons

Pros:

1. No change on all existing APIs

Cons:

1. Only provide coarse-grained control on all graphs
2. Need efforts for managing global status.
3. Compared with option 2, cannot provide different fusion strategies if needed.

### Option 2: Support accumulation mode on graph level( Recommended )

All the computations or operations in a graph will be specified with the same
accumulation data type. Whether the given accumulation mode can be utilized is
decided by the backend capbility. Users will be required to provide the
attribute value while creating the graph.

This option refers to the current solution for floating-point match mode
attribute support in oneDNN Graph. These attributes share some similarities to
a certain extent: both provide a certain degree of speedup by modifying the
datatype based on backend capabilities.

Since we already have the `fpmath-mode` attribute, in order to make the API 
simple and scalable, we will need to introduce a new API, `graph config`, to 
wrap all the attributes for graph the constructor. Users can now create a
`config` with specific attributes or set attribute values later. This new
design maintains compatibility with current APIs while offers improved
flexibility.

The API will be like:

```cpp
dnnl::graph::graph::config cfg;
cfg.set_engine_kind(dnnl::engine::kind::cpu);
cfg.set_fpmath_mode(dnnl::fpmath_mode::strict);
cfg.set_accumulation_mode(dnnl::accumulation_mode::strict);

graph g( dnnl::engine::kind::cpu, cfg);
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

The current graph ctor API will remain unchanged but might be deprecated in the
future, hence it's recommended to use `graph config` to create a graph object
if the attributes are needed. 

#### Pros and Cons

Pros:

1. No changes are needed for op-related and partition-related operations, such
as op creation and compilation.
2. Graph may provide different fusion strategies based on the given accumulation
mode according to backend capabilities.
3. Easier usage. Users only need to set once for a certain graph.
4. Compared with option 1,  can be more useful by allowing users to create
multiple graphs with different modes.


Cons:

1. Relatively coarse-grained control. 

### Option 3: Support accumulation mode on partition level

Users will be asked to provide the accumulation mode during compilation stage.
In this case, users may need to set the accumulation mode for different
partitions separately. Users are not required to provide the attribute util the
compilation stage.

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

1. Users may specify different accumulation mode for different backends.
2. Users may compile the same partition with different accumulation modes.

Cons:

1. Compared with option 2, the library cannot provide different fusion
strategies if needed.

### Option 4: Support accumulation mode on op level

Users will be asked to set the accumulation mode as an attribute when creating
an op. The attribute will be available for conv/deconv/matmul ops. The API will
be like:

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

Cons:

1. More integration code change compared with other options.
2. Can result in complicated fusion logic if the ops in a subgraph use
different modes.

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
  "fpmath_mode": "strict",
  "accumulation_mode": "strict",
  "graph": [ 
    //... 
  ]
}
```

For global API or partition-level API, benchdnn graph needs to be updated and
the input should be given with command-line knobs.

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
