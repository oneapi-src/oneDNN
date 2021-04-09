# Getting started with programming model {#get_started_with_programming}

## Description

Previously, oneDNN Graph assumes that users have a graph representation and a
graph executor which drives the overall execution. That means, oneDNN Graph
supports constructing a graph by users, but has limited support for users'
program to directly compile and execute the returned partitions.

Now, oneDNN Graph features the support of minimum programming model.
Users can easily construct a self-defined graph and generate the corresponding
partitions. After that, users can compile and execute those partitions.

Here, an example will be provided to show the programming model. The full
example code can be found at
[cpu_programming.cpp](../examples/cpp/src/cpu_programming.cpp).

## cpu_programming_tutorial() function

### Create tensor mapping

In order to provide a running context, a new
class named [tensor_map](../examples/cpp/include/common/execution_context.hpp#L36)
is being introduced in this example. A tensor map will be responsible for
holding all the tensors that will be used in the users' program. it contains the
mapping from an unique logical tensor id to the corresponding tensor.

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
[dnnl::graph::logical_tensor](../include/oneapi/dnnl/dnnl_graph.hpp#L290) is
used to express the connection relationship between different ops. So for the
first `Convolution` op, users can construct all input and output logical tensors
like below.

~~~cpp
logical_tensor conv_data_lt {id_mgr["conv_data"], data_type::f32, input_dims, layout_type::strided};
logical_tensor conv_weight_lt {id_mgr["conv_weight"], data_type::f32, weight_dims, layout_type::strided};
logical_tensor conv_bias_lt {id_mgr["conv_bias"], data_type::f32, bias_dims, layout_type::strided};
logical_tensor conv_dst_lt {id_mgr["dst_dims"], data_type::f32, dst_dims, layout_type::strided};
~~~

Here [`id_mgr`](../examples/cpp/include/common/utils.hpp#120) is a utility class
to generate unique id according to the given name. It requires the 1:1 mapping
between id and the given name.

**Note**: These examples create logical tensors with complete shape information
and use them in the partition compilation. Currently, the library also supports
an API to infer the output shapes according to input shapes. We don't leverage
this API here as it's planned to be deprecated in the future so the compilation
can accept incomplete output shapes.

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
[`get_partitions()`](../include/oneapi/dnnl/dnnl_graph.hpp#L1274), users can get
several partitions in topological order.

~~~cpp
std::vector<partition> partitions = g.get_partitions();
~~~

### Compile partitions and execute

In the real workload, users need to provide the device information to compile a
partition. Typically, a
[dnnl::graph::engine](../include/oneapi/dnnl/dnnl_graph.hpp#L97) should be
created with an engine kind and a device id. The engine kind should be the same
as the one used to create the graph.

~~~cpp
int device_id = 0;
engine e {engine_kind, device_id};
~~~

In oneDNN Graph, a
[dnnl::graph::stream](../include/oneapi/dnnl/dnnl_graph.hpp#L239) is the logical
abstraction for execution units. It is created on top of oneDNN Graph engine.

~~~cpp
stream s {e};
~~~

Currently, a partition also has a flag to indicate the supporting status. If the
flag is True, that partition is supported by oneDNN Graph backend. Otherwise,
that partition is not supported by oneDNN Graph backend and users need to handle
the computation by themselves.

- [`is_supported()`](../include/oneapi/dnnl/dnnl_graph.hpp#L790)

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
std::vector<tensor> input_ts = construct_and_initialize_tensors(inputs, c_partitions[i], tm, 1);
std::vector<tensor> output_ts = construct_and_initialize_tensors(outputs, c_partitions[i], tm, 0);
~~~

Finally, users can execute the compiled partition with input and output tensors.

~~~cpp
c_partitions[i].execute(s, input_ts, output_ts);
~~~

After finishing executing all compiled partitions, users can get the final
results of the graph.
