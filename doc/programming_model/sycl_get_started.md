# Getting started on both CPU and GPU with SYCL extensions API {#dev_guide_sycl_programming}

@warning This tutorial is deprecated due to the programming model changes and
will be removed soon, please refer to
[sycl_simple_pattern.cpp](../../examples/cpp/src/sycl_simple_pattern.cpp) for
latest version.

This is an example to demonstrate how to build a simple graph and run on CPU/GPU
with SYCL extension APIs.

In this example, you will learn the below things about oneDNN Graph.

- How to build a graph and get several partitions
- How to create engine, allocator and stream
- How to compile a partition
- How to execute a compiled partition with input tensors on a specific stream

**Note**: currently, oneDNN Graph has limited support for direct programming
model. For example, users need to know how many inputs/outputs of a self-defined
pattern or partition.

The full example code can be found at
[sycl_get_started.cpp](../../examples/cpp/src/sycl_get_started.cpp).

## Public headers

To start using oneDNN Graph, users should include the
[dnnl_graph.hpp](../../include/oneapi/dnnl/dnnl_graph.hpp) header file in the
application. If a user wants to run on a SYCL device like this example, he/she
also need include
[dnnl_graph_sycl.hpp](../../include/oneapi/dnnl/dnnl_graph_sycl.hpp). All the
C++ APIs reside in namespace `dnnl::graph`.

~~~cpp
#include "oneapi/dnnl/dnnl_graph.hpp"
#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
using namespace dnnl::graph;
using namespace cl::sycl; // for SYCL related APIs
~~~

## sycl_get_started_tutorial() function

### Build graph and get partitions

In this section, firstly we will build a graph containing the pattern like
`conv0->relu0->conv1->relu1`. After that, we can get all of partitions which are
determined by backend.

To create a graph,
[`dnnl::graph::engine::kind`](../../include/oneapi/dnnl/dnnl_graph.hpp#L102) is
needed because the returned partitions may vary on different devices.

~~~cpp
graph g(engine_kind);
~~~

To build a graph, the connection relationship of different ops must be known. In
oneDNN Graph, the `id` of
[`dnnl::graph::logical_tensor`](../../include/oneapi/dnnl/dnnl_graph.hpp#L290)
is used to express such relationship. Besides that, a logical tensor describes
the metadata of a tensor, like element data type, number of dimensions, size for
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
find the definitions in our [public
specification](https://spec.oneapi.com/onednn-graph/latest/ops/convolution/Convolution_1.html).

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

For the first `BiasAdd` and `ReLU` operators, one can use the similar way to
create them.

~~~cpp
logical_tensor conv0_bias_desc {logical_id[3], logical_tensor::data_type::f32, bias_dims, logical_tensor::layout_type::undef};
logical_tensor conv0_bias_add_dst_desc {logical_id[4], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::undef};
op conv0_bias_add(1, op::kind::BiasAdd, {conv0_dst_desc, conv0_bias_desc}, {conv0_bias_add_dst_desc}, "conv0_bias_add");
logical_tensor relu0_dst_desc {logical_id[5], logical_tensor::data_type::f32, dst_dims, logical_tensor::layout_type::undef};
op relu0(2, op::kind::ReLU, {conv0_bias_add_dst_desc}, {relu0_dst_desc}, "relu0");
~~~

The creations of rest operators is as similar as above, please find the [example
code](../../examples/cpp/src/sycl_get_started.cpp#L141) for details.

After all those operators are created, users can add these ops into the graph.

~~~cpp
g.add_op(conv0);
g.add_op(relu0);
g.add_op(conv1);
g.add_op(relu1);
g.add_op(conv0_bias_add);
g.add_op(conv1_bias_add);
~~~

Then by calling
[`get_partitions()`](../../include/oneapi/dnnl/dnnl_graph.hpp#L1287), users can
get several partitions. In this example, there should be two partitions:
`conv0+relu0` and `conv1+relu1`.

**Note**: setting env variable `DNNL_GRAPH_DUMP=1` can save internal graphs into
dot files before/after graph fusion.

~~~cpp
auto partitions = g.get_partitions();
~~~

### Compile partitions

In the real world, frameworks will provide device info for oneDNN Graph. For
example, they may provide a `sycl::queue` like below:

~~~cpp
sycl::queue q = (engine_kind == engine::kind::gpu) ? sycl::queue(gpu_selector {}) : sycl::queue(cpu_selector {});
~~~

Based on the above `sycl::queue`, users can create a
[dnnl::graph::allocator](../../include/oneapi/dnnl/dnnl_graph.hpp#L45) and
[dnnl::graph::engine](../../include/oneapi/dnnl/dnnl_graph.hpp#L97). Here,
`sycl_malloc_wrapper` and `sycl_free_wrapper` are call-back functions and also
provided by frameworks.

In oneDNN Graph, SYCL extension APIs reside in the namespace
`dnnl::graph::sycl_interop`, which are defined at
[dnnl_graph_sycl.hpp](../../include/oneapi/dnnl/dnnl_graph_sycl.hpp).

~~~cpp
engine eng = sycl_interop::make_engine(q.get_device(), q.get_context());
allocator alloc = sycl_interop::make_allocator(sycl_malloc_wrapper, sycl_free_wrapper);
eng.set_allocator(alloc);
~~~

At runtime, the information about a tensor should be known, including `shape`
and `layout`. So here users need to create those input and output logical
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
logical_tensor relu1_dst_desc_plain {logical_id[10], logical_tensor::data_type::f32, dst1_dims, logical_tensor::layout_type::strided};

auto cp1 = partitions[1].compile({relu0_dst_desc_plain, conv1_weight_desc_plain, conv1_bias_desc_plain}, {relu1_dst_desc_plain}, eng);
~~~

### Execute the compiled partitions

In oneDNN Graph, a
[dnnl::grap::stream](../../include/oneapi/dnnl/dnnl_graph.hpp#L239) is the
logical abstraction for execution units. It is created on top of oneDNN Graph
engine. For SYCL device, it also contains an opencl queue.

~~~cpp
auto strm = sycl_interop::make_stream(eng, q);
~~~

Since this example is to show how to run on a SYCL device, users need to prepare
SYCL memory buffer firstly. In the real world, these buffers are probably
already prepared by frameworks. Here, we use
[`malloc_shared`](https://docs.oneapi.com/versions/latest/dpcpp/iface/usm-malloc.html#sycl-malloc-shared)
to request USM buffers which is an interface provided by DPCPP.

~~~cpp
auto conv0_src_data = (float *)malloc_shared(static_cast<size_t>(product(input_dims)) * sizeof(float), q.get_device(), q.get_context());
auto conv0_weight_data = (float *)malloc_shared(static_cast<size_t>(product(weight_dims)) * sizeof(float), q.get_device(), q.get_context());
auto conv0_bias_data = (float *)malloc_shared(static_cast<size_t>(product(bias_dims)) * sizeof(float), q.get_device(), q.get_context());
auto relu0_dst_data = (float *)malloc_shared(static_cast<size_t>(product(dst_dims)) * sizeof(float), q.get_device(), q.get_context());
auto conv1_weight_data = (float *)malloc_shared(static_cast<size_t>(product(weight1_dims)) * sizeof(float), q.get_device(), q.get_context());
auto conv1_bias_data = (float *)malloc_shared(static_cast<size_t>(product(bias1_dims)) * sizeof(float), q.get_device(), q.get_context());
auto relu1_dst_data = (float *)malloc_shared(static_cast<size_t>(product(dst1_dims)) * sizeof(float), q.get_device(), q.get_context());
~~~

Before the execution, users also need to bind the logical tensor and memory
buffer to [dnnl::graph::tensor](../../include/oneapi/dnnl/dnnl_graph.hpp#L542).

~~~cpp
tensor conv0_src(conv0_src_desc_plain, eng, conv0_src_data);
tensor conv0_weight(conv0_weight_desc_plain, eng, conv0_weight_data);
tensor conv0_bias(conv0_bias_desc_plain, eng, conv0_bias_data);
tensor relu0_dst(relu0_dst_desc_plain, eng, relu0_dst_data);
std::vector<tensor> out0_list = {relu0_dst};

sycl_interop::execute(cp0, strm, {conv0_src, conv0_weight, conv0_bias}, out0_list);
~~~

In the same way, users can execute the second compiled partition.

~~~cpp
tensor conv1_weight(conv1_weight_desc_plain, eng, conv1_weight_data);
tensor conv1_bias(conv1_bias_desc_plain, eng, conv1_bias_data);
tensor relu1_dst(relu1_dst_desc_plain, eng, relu1_dst_data);
std::vector<tensor> out1_list {relu1_dst};

sycl_interop::execute(cp1, strm, {relu0_dst, conv1_weight, conv1_bias}, out1_list);
~~~
