# Getting Started with Programming Model {#dev_guide_cpu_programming}

## Description

Previously, oneDNN Graph assumes that users have a graph representation and a
graph executor which drives the overall execution. That means, oneDNN Graph
supports constructing a graph by users, but has limited support for users'
program to directly compile and execute the returned partitions.

Now, oneDNN Graph features the support of minimum programming model. Users can
easily construct a self-defined graph and generate the corresponding partitions.
After that, users can compile and execute those partitions.

Here, an example will be provided to show the programming model. The full
example code can be found at
[cpu_programming.cpp](../../tests/demo/src/cpu_programming.cpp).

## cpu_programming_tutorial() function

### Create tensor mapping

In order to provide a running context, a new class named
[tensor_map](../../tests/demo/include/common/execution_context.hpp#L51) is
being introduced in this example. A tensor map will be responsible for holding
all the tensors that will be used in the users' program. it contains the mapping
from an unique logical tensor id to the corresponding tensor.

- `std::unordered_map<size_t, tensor> data_`

Before building graph, users should create such a tensor map like below:

~~~cpp
tensor_map tm;
~~~

### Build graph and get partitions

First of all, a `graph` is needed since it will help users to construct the
self-defined graph.

~~~cpp
graph g(engine_kind);
~~~

In this example, the below graph will be used. It contains `Convolution`,
`Wildcard`, `Add`, `ReLU` and `End` ops.

~~~markdown
      Convolution Wildcard
           |         |
        tensor1   tensor2
       /      \     /
     End        Add
                 |
              tensor3
                 |
                ReLU
                 |
              tensor4
                 |
                End
~~~

In oneDNN Graph, the id of
[dnnl::graph::logical_tensor](../../include/oneapi/dnnl/dnnl_graph.hpp#L365) is
used to express the connection relationship between different ops. So for the
first `Convolution` op, users can construct all input and output logical tensors
like below.

~~~cpp
logical_tensor conv_data_lt {id_mgr["conv_data"], data_type::f32, input_dims, layout_type::strided};
logical_tensor conv_weight_lt {id_mgr["conv_weight"], data_type::f32, weight_dims, layout_type::strided};
logical_tensor conv_bias_lt {id_mgr["conv_bias"], data_type::f32, bias_dims, layout_type::strided};
logical_tensor conv_dst_lt {id_mgr["dst_dims"], data_type::f32, dst_dims, layout_type::strided};
~~~

Here [`id_mgr`](../../tests/demo/include/common/utils.hpp#245) is a utility
class to generate unique id according to the given name. It requires the 1:1
mapping between id and the given name.

**Note**: These examples create logical tensors with complete shape information
and use them in the partition compilation. Currently, oneDNN Graph also supports
the output logical tensors with incomplete shape information (containing -1).
oneDNN Graph implementation will calculate the output shapes according to given
input shapes and schema of the OP. After compilation finished, users can query
the compiled partition for the output logical tensors and get the shapes.

Next step is to create a `Convolution` op with the above inputs and outputs.

~~~cpp
op conv {0, op::kind::Convolution, {conv_data_lt, conv_weight_lt, conv_bias_lt}, {conv_dst_lt}, "conv_0"};
conv.set_attr<std::vector<int64_t>>("strides", {1, 1});
conv.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
conv.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
conv.set_attr<std::vector<int64_t>>("dilations", {1, 1});
conv.set_attr<int64_t>("groups", 1);
conv.set_attr<std::string>("data_format", "NCX");
conv.set_attr<std::string>("filter_format", "OIX");
~~~

The similar way can be used to construct other logical tensors and ops.

~~~cpp
logical_tensor add_input_2_lt {id_mgr["add_input_2"], data_type::f32, add_input_2_dims, layout_type::strided};
logical_tensor add_dst_lt {id_mgr["add_dst"], data_type::f32, dst_dims, layout_type::strided};

op add {1, op::kind::Add, {conv_dst_lt, add_input_2_lt}, {add_dst_lt}, "add_0"};

logical_tensor relu_dst_lt {id_mgr["relu_dst"], data_type::f32, dst_dims, layout_type::strided};

op relu {2, op::kind::ReLU, {add_dst_lt}, {relu_dst_lt}, "relu_0"};

op wildcard {3, op::kind::Wildcard, {}, {add_input_2_lt}, "wildcard_0"};
~~~

Users are also able to specify the output(s) of the graph through `End` op.
Below code is used to specify the outputs of `Convolution` and `ReLU` ops, which
are also the outputs of the whole graph.

~~~cpp
op end_0 {4, op::kind::End, {conv_dst_lt}, {}, "end_0"};
op end_1 {5, op::kind::End, {relu_dst_lt}, {}, "end_1"};
~~~

After all those ops are created, users can add these ops into the graph.

~~~cpp
g.add_op(conv);
g.add_op(add);
g.add_op(relu);
g.add_op(wildcard);
g.add_op(end_0);
g.add_op(end_1);
~~~

Then by calling
[`get_partitions()`](../../include/oneapi/dnnl/dnnl_graph.hpp#L1894), users can get
several partitions in topological order.

~~~cpp
std::vector<partition> partitions = g.get_partitions();
~~~

### Compile partitions and execute

In the real workload, users need to provide the device information to compile a
partition. Typically, a
[dnnl::graph::engine](../../include/oneapi/dnnl/dnnl_graph.hpp#L254) should be
created with an engine kind and a device id. The engine kind should be the same
as the one used to create the graph.

~~~cpp
engine e {engine_kind, 0};
~~~

In oneDNN Graph, a
[dnnl::graph::stream](../../include/oneapi/dnnl/dnnl_graph.hpp#L318) is the
logical abstraction for execution units. It is created on top of oneDNN Graph
engine.

~~~cpp
stream s {e};
~~~

Currently, a partition also has a flag to indicate the supporting status. If the
flag is True, that partition is supported by oneDNN Graph backend. Otherwise,
that partition is not supported by oneDNN Graph backend and users need to handle
the computation by themselves.

- [`is_supported()`](../../include/oneapi/dnnl/dnnl_graph.hpp#L1707)

~~~cpp
// create the vector to store all compiled partitions
std::vector<compiled_partition> c_partitions(partitions.size());
// compilation-execution loop
for (size_t i = 0; i < partitions.size(); ++i) {
    if (partitions[i].is_supported()) {
        // 1. get inputs and outputs from a partition
        // 2. compile the partition to generate a compile partition
        // 3. construct tensors with logical tensors and allocated memory buffer
        // 4. execute the compiled partition with the stream
    } else {
        // users need to write code to compute this partition
    }
}
~~~

At the compilation stage, users need to provide input and output logical tensors
for oneDNN Graph compilation API. The below APIs will be used for this purpose.

- `partition.get_in_ports()`: return the list of input logical tensors from the
  partition

- `partition.get_out_ports()`: return the list of output logical tensors from
  the partition

~~~cpp
std::vector<logical_tensor> inputs = partitions[i].get_in_ports();
std::vector<logical_tensor> outputs = partitions[i].get_out_ports();
~~~

Then, users can compile the partition according to the given inputs/outputs.

~~~cpp
c_partitions[i] = partitions[i].compile(inputs, outputs, e);
~~~

Before executing, users need to construct input and output tensors with memory
buffer. So, a helper function is provided in the example like below:

- `construct_and_initialize_tensors()`: construct and initialize tensors
  according to the given logical tensors.

~~~cpp
std::vector<tensor> input_ts = construct_and_initialize_tensors(inputs, c_partitions[i], tm, e, 1);
std::vector<tensor> output_ts = construct_and_initialize_tensors(outputs, c_partitions[i], tm, e, 0);
~~~

Finally, users can execute the compiled partition with input and output tensors.

~~~cpp
c_partitions[i].execute(s, input_ts, output_ts);
~~~

After finishing executing all compiled partitions, users can get the final
results of the graph.

## Single Operator Partition

In order to simplify the support of framework imperative execution mode, oneDNN
Graph API supports single op partition. As the name suggests, it's a partition
which only contains one operator. There is no need to use graph for imperative
execution mode. The demo code is like below. The full example code can be found
at
[cpu_single_op_partition_f32.cpp](../../examples/cpp/cpu_single_op_partition_f32.cpp).

~~~cpp
// define input and output logical tensor
logical_tensor lt0 {0, data_type::f32, {1, 3, 256, 256}, layout_type::strided}; // 1
logical_tensor lt1 {1, data_type::f32, {32, 3, 3, 3}, layout_type::strided};    // 2
logical_tensor lt2 {2, data_type::f32, {1, 32, 252, 252}, layout_type::strided};// 3

// step 1: create the op
op conv {0, kind::Convolution, {lt0, lt1}, {lt2}, "convolution"};               // 4
// set many attributes
conv.set_attr("pads_begin", {0,0});                                             // 5
conv.set_attr("pads_end", {0,0});                                               // 6

// step 2: create partition with the given operator
partition part {conv, engine_kind};                                             // 7

// step 3: compile the partition
engine eng {engine_kind, 0};                                                    // 8
compiled_partition cpart = part.compile({lt0, lt1}, {lt2}, eng);                // 9

// step 4: execute the compiled partition
tensor data {lt0, eng, buf0};                                                   // 10
tensor weight {lt1, eng, buf1};                                                 // 11
tensor output {lt2, eng, buf2};                                                 // 12
cp.execute(stream, {data, weight}, {output});                                   // 13
~~~

## Supporting Fall-back Mechanism for Users

Sometimes framework users may want to fall back to default implementation when
oneDNN Graph fails to compile or execute. At this time users need to reorder
oneDNN Graph opaque tensors to public tensors and then feed into framework
default implementation kernel. oneDNN Graph API supports `reorder` operator for
this case. With the help of feature of single operator partition, users can
easily convert tensors to public layout.

## Weight Prepacking

For inference mode, weights are usually constant during iterations. In order to
improve inference performance, users may want to convert the weight from public
layout to opaque layout and use this converted weight every iteration. This will
significantly reduce the overhead of converting weight every iteration. With the
combination of single operator partition and `reorder` operation, users are able
to implement this optimization on their side. The weight's opaque layout can be
only queried out from compiled partition, which requires the tensor shapes must
be known at the compilation time.

## Floating-point Math Mode

Floating-point math mode is used to specify whether implicit down conversion
from f32 to those compatible floating data types (bf16/tf32/f16) is allowed. By
specifying floating-point math mode, users may see the performance improvement (
depending on library backend capability) without explicitly modifying the
original f32 models. oneDNN Graph supports setting floating-point math mode
through `graph` constructor or by using the environment variable
`DNNL_DEFAULT_FPMATH_MODE` (require no code changes).

~~~cpp
// define input and output logical tensor
logical_tensor conv_src {0, data_type::f32, {1, 3, 256, 256}, layout_type::strided}; // 1
logical_tensor conv_wei {1, data_type::f32, {32, 3, 3, 3}, layout_type::strided};    // 2
logical_tensor conv_dst {2, data_type::f32, {1, 32, 252, 252}, layout_type::strided};// 3
logical_tensor relu_dst {3, data_type::f32, {1, 32, 252, 252}, layout_type::strided};// 4

// step 1: create the op
op conv {0, kind::Convolution, {conv_src, conv_wei}, {conv_dst}, "convolution"};     // 5
// set many attributes
conv.set_attr("pads_begin", {0,0});                                                  // 6
conv.set_attr("pads_end", {0,0});                                                    // 7

op relu {1, kind::ReLU, {conv_dst}, {relu_dst}, "relu"}                              // 8

engine::kind engine_kind = engine::kind::cpu;                                        // 9

// step 2: create graph and add ops to graph
graph g(engine_kind, graph::fpmath_mode::tf32);                                      // 10 <---- specify fpmath mode via explicit API

g.add_op(conv);                                                                      // 11
g.add_op(relu);                                                                      // 12

// step 3: get partitions
// all returned partition from the graph will share the same fpmath mode
std::vector<partition> parts = g.get_partitions();                                   // 13

// step 3: compile the partition
engine eng {engine_kind, 0};                                                         // 14
compiled_partition cpart = parts[0].compile({conv_src, conv_wei}, {relu_dst}, eng);  // 15

// step 4: execute the compiled partition
tensor src {conv_src, eng, buf0};                                                    // 16
tensor wei {conv_wei, eng, buf1};                                                    // 17
tensor dst {relu_dst, eng, buf2};                                                    // 18
// the implicit down conversion from f32 to tf32 may happen and then the
// underlying kernel will compute on tf32 data type instead of f32.
cp.execute(stream, {src, wei}, {dst});                                               // 19
~~~

## Additional Ease-of-Use Features

- Users are not required to set the tensors' format

  - oneDNN Graph API removes the format representation from logical tensor, so
    users don't need to indicate whether the tensor is NHWC tensor or NCHW
    tensor. Instead, the logical tensor used in oneDNN Graph API only needs to
    describe the layout using dims and strides. The semantics of axes for
    tensors are specified by the operator's attributes. For example, the
    Convolution operator has `data_format` and `filter_format`.

  ~~~cpp
  // for example, tensorflow shapes
  dims_t ish = {1, 227, 227, 3};
  dims_t wsh = {11, 11, 3, 96};
  dims_t osh = {1, 55, 55, 96};
  // create input/output logical tensors.
  logical_tensor conv_src {0, dt::f32, ish, layout_type::strided};
  logical_tensor conv_wei {1, dt::f32, wsh, layout_type::strided};
  logical_tensor conv_dst {2, dt::f32, osh, layout_type::strided};
  // create convolution op with inputs and outputs
  op conv {0, kind::Convolution, {conv_src, conv_wei}, {conv_dst}, “conv0”};
  // set attributes to conv op
  conv.set_attr("data_format", std::string("NXC")); // input, nhwc
  conv.set_attr("filter_format", std::string("XIO")); // weight, hwio
  // add op to graph.
  graph.add_op(conv);
  ~~~

- Users are not required to "query and convert" to the opaque layout for input
  tensors at execution time

  - oneDNN Graph API allows that the output tensor of one partition is
    directly passed to another partition, even though this output tensor has
    opaque layout.

  ~~~cpp
  // create logical tensors, previous layer also runs oneDNN Graph partition
  logical_tensor conv_src {0, dt::f32, {1, 3, 227, 227}, layout_type::opaque};
  logical_tensor conv_wei {1, dt::f32, {96, 3, 11, 11}, layout_type::strided};
  logical_tensor conv_dst {2, dt::f32, {-1, -1, -1, -1}, layout_type::any};
  // create convolution op with inputs and outputs
  op conv {0, kind::Convolution, {conv_src, conv_wei}, {conv_dst}, “conv0”};
  // add op to graph.
  graph.add_op(conv);
  // get partitions with debug policy.
  std::vector<partition> partitions = graph.get_partitions();
  // compile the first partition and set blocked format to output tensor
  compiled_partition cp = partitions[0].compile({conv_src, conv_wei}, {conv_dst}, eng);
  // execute the compiled partition with input/output tensors
  tensor src_tensor = tensor_from_last_layer; // with opaque layout id
  tensor wei_tensor = tensor(conv_wei, eng, buf_wei); // strided weight
  tensor dst = tensor(cp.query_logical_tensor(2), eng, buf_dst);
  cp.execute(stream, {src_tensor, wei_tensor}, {dst_tensor});
  ~~~
