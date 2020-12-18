# Getting started on CPU {#get_started_with_cpu}

This is an example to demonstrate how to build a simple graph and run on CPU.

In this example, you will learn the below things about oneDNN graph.

- How to build a graph and get several partitions
- How to create engine, allocator and stream
- How to compile a partition
- How to execute a compiled partition with input tensors on a specific stream

**Note**: currently, oneDNN graph has limited support for direct programming
model. For example, users need to know how many inputs/outputs of a self-defined
pattern or partition.

The full example code can be found at [cpu_get_started.cpp](../examples/cpp/src/cpu_get_started.cpp).

## Public headers

To start using oneDNN graph, users should include the
[dnnl_graph.hpp](../include/oneapi/dnnl/dnnl_graph.hpp) header file in the
application. All the C++ APIs reside in namespace `dnnl::graph`.

~~~cpp
#include "oneapi/dnnl/dnnl_graph.hpp"
using namespace dnnl::graph;
~~~

## cpu_get_started_tutorial() function

### Build graph and get partitions

In this section, firstly we will build a graph containing the pattern like
`conv0->relu0->conv1->relu1`. After that, we can get all of partitions which are
determined by backend.

To create a graph,
[`dnnl::graph::engine::kind`](../include/oneapi/dnnl/dnnl_graph.hpp#L102) is
needed because the returned partitions may vary on different devices.

~~~cpp
graph g(engine_kind);
~~~

To build a graph, the connection relationship of different ops must be known. In
oneDNN graph, the `id` of
[`dnnl::graph::logical_tensor`](../include/oneapi/dnnl/dnnl_graph.hpp#L282) is
used to express such relationship. Besides that, a logical tensor describes the
metadata of a tensor, like element data type, number of dimensions, size for
each dimension (shape), layout, and the total size of the data.

So for the first `Convolution` operator, input and output logical tensors are
created as below. At this stage, the information in a logical tensor maybe not
complete. For example, `shape` or `layout` is still unknown at the early graph
optimization pass.

~~~cpp
logical_tensor conv0_src_desc {logical_id[0], logical_tensor::data_type::f32, input_dims, logical_tensor::layout_type::undef};
logical_tensor conv0_weight_desc {logical_id[1], logical_tensor::data_type::f32, weight_dims,logical_tensor::layout_type::undef};
logical_tensor conv0_dst_desc {logical_id[2], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::undef};
~~~

Next step is to create a `Convolution` operator with all of input/output logical
tensors and required attributes. For more details about these attributes, please
find the definitions in our
[public specification](https://spec.oneapi.com/onednn-graph/latest/ops/convolution/Convolution_1.html).

~~~cpp
op conv0(0, op::kind::Convolution, {conv0_src_desc, conv0_weight_desc}, {conv0_dst_desc}, "conv0");
conv0.set_attr<std::vector<int64_t>>("strides", {4, 4});
conv0.set_attr<std::vector<int64_t>>("pads_begin", {0, 0});
conv0.set_attr<std::vector<int64_t>>("pads_end", {0, 0});
conv0.set_attr<std::vector<int64_t>>("dilations", {1, 1});
conv0.set_attr<int64_t>("groups", 1);
conv0.set_attr<std::string>("data_format", "NCX");
conv0.set_attr<std::string>("filter_format", "OIX");
~~~

For the first `BiasAdd` and `ReLU` operators, users can use the similar way to
create them.

~~~cpp
logical_tensor conv0_bias_desc {logical_id[3], logical_tensor::data_type::f32, bias_dims, logical_tensor::layout_type::undef};
logical_tensor conv0_bias_add_dst_desc {logical_id[4], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::undef};
op conv0_bias_add(1, op::kind::BiasAdd, {conv0_dst_desc, conv0_bias_desc}, {conv0_bias_add_dst_desc}, "conv0_bias_add");
logical_tensor relu0_dst_desc {logical_id[5], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::undef};
op relu0(2, op::kind::ReLU, {conv0_bias_add_dst_desc}, {relu0_dst_desc}, "relu0");
~~~

The creations of rest operators is as similar as above, please find the
[example code](../examples/cpp/src/cpu_get_started.cpp#L141) for details.

After all those operators are created, users can add these ops into the graph.

~~~cpp
g.add_op(conv0);
g.add_op(relu0);
g.add_op(conv1);
g.add_op(relu1);
g.add_op(conv0_bias_add);
g.add_op(conv1_bias_add);
~~~

Then by calling [`get_partitions()`](../include/oneapi/dnnl/dnnl_graph.hpp#L1200),
users can get several partitions. In this example, there should be two
partitions: `conv0+relu0` and `conv1+relu1`.

**Note**: setting env variable `DNNL_GRAPH_DUMP=1` can save internal graphs into
dot files before/after graph fusion.

~~~cpp
auto partitions = g.get_partitions();
~~~

### Compile partitions

In the real world, frameworks will provide device info for oneDNN graph. For
example, they may provide a `device_id` like below:

~~~cpp
const int32_t device_id = 0;
~~~

Based on the above `device_id`, users can create a
[dnnl::graph::engine](../include/oneapi/dnnl/dnnl_graph.hpp#L97). In this
example, a default [dnnl::graph::allocator](../include/oneapi/dnnl/dnnl_graph.hpp#L45)
is used. Users can also set a self-defined allocator to the engine.

~~~cpp
engine eng {engine_kind, device_id};
allocator alloc {};
eng.set_allocator(alloc);
~~~

At runtime, the information about a tensor should be known, including `shape`
and `layout`, so here users need to create those input and output logical
tensors again with concrete info.

~~~cpp
logical_tensor conv0_src_desc_plain {logical_id[0], logical_tensor::data_type::f32, input_dims, logical_tensor::layout_type::strided};
logical_tensor conv0_weight_desc_plain {logical_id[1], logical_tensor::data_type::f32, weight_dims, logical_tensor::layout_type::strided};
logical_tensor conv0_bias_desc_plain {logical_id[3], logical_tensor::data_type::f32, bias_dims, logical_tensor::layout_type::strided};
logical_tensor relu0_dst_desc_plain {logical_id[5], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::strided};
~~~

Then compile the partition 0 to get compiled partition.

~~~cpp
auto cp0 = partitions[0].compile({conv0_src_desc_plain, conv0_weight_desc_plain, conv0_bias_desc_plain}, {relu0_dst_desc_plain}, eng);
~~~

In the same way, users can get the compiled partition from the partition 1.

~~~cpp
logical_tensor conv1_weight_desc_plain {logical_id[6], logical_tensor::data_type::f32, weight1_dims, logical_tensor::layout_type::strided};
logical_tensor conv1_bias_desc_plain {logical_id[8], logical_tensor::data_type::f32, bias1_dims, logical_tensor::layout_type::strided};
logical_tensor relu1_dst_desc_plain {logical_id[10], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::strided};

auto cp1 = partitions[1].compile({relu0_dst_desc_plain, conv1_weight_desc_plain, conv1_bias_desc_plain}, {relu1_dst_desc_plain}, eng);
~~~

### Execute the compiled partitions

In oneDNN graph, a [dnnl::grap::stream](../include/oneapi/dnnl/dnnl_graph.hpp#L239)
is the logical abstraction for execution units. It is created on top of oneDNN
graph engine.

~~~cpp
stream strm {eng};
~~~

The below code is to prepare memory buffers. In the real world, these buffers
are probably already prepared by frameworks.

~~~cpp
std::vector<float> conv0_src_data(static_cast<size_t>(product(input_dims)), 1.0f);
std::vector<float> conv0_weight_data(static_cast<size_t>(product(weight_dims)), 1.0f);
std::vector<float> conv0_bias_data(static_cast<size_t>(product(bias_dims)), 1.0f);
std::vector<float> relu0_dst_data(static_cast<size_t>(product(dst_dims)), 0.0f);
std::vector<float> conv1_weight_data(static_cast<size_t>(product(weight1_dims)), 1.0f);
std::vector<float> conv1_bias_data(static_cast<size_t>(product(bias1_dims)), 1.0f);
std::vector<float> relu1_dst_data(static_cast<size_t>(product(dst1_dims)), 0.0f);
~~~

Before the execution, users also need to bind the logical tensor and memory
buffer to [dnnl::graph::tensor](../include/oneapi/dnnl/dnnl_graph.hpp#L530).

~~~cpp
tensor conv0_src(conv0_src_desc_plain, conv0_src_data.data());
tensor conv0_weight(conv0_weight_desc_plain, conv0_weight_data.data());
tensor conv0_bias(conv0_bias_desc_plain, conv0_bias_data.data());
tensor relu0_dst(relu0_dst_desc_plain, relu0_dst_data.data());

cp0.execute(strm, {conv0_src, conv0_weight, conv0_bias}, {relu0_dst});
~~~

In the same way, users can execute the second compiled partition.

~~~cpp
tensor conv1_weight(conv1_weight_desc_plain, conv1_weight_data.data());
tensor conv1_bias(conv1_bias_desc_plain, conv1_bias_data.data());
tensor relu1_dst(relu1_dst_desc_plain, relu1_dst_data.data());

cp1.execute(strm, {relu0_dst, conv1_weight, conv1_bias}, {relu1_dst});
~~~
