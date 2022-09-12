# oneDNN Graph API Concepts {#dev_guide_graph_basic_concepts}

## Introduction

oneDNN Graph API programming model allows users to express their computational
graph and generate optimized sub-graphs called Partitions. Partitions are
decided by oneDNN Graph API implementation, which is the key concept to satisfy
the different needs of AI hardware classes using a unified API. Users then
compile partitions, bind tensor data, and execute compiled partitions. The key
concepts include *logical tensor*, *op*, *graph*, *partition*,
*compiled partition* and *tensor*. Here is the relationship between these
entities.

@img{img_graph_programming_model.jpg,Figure 1: Overview of Graph API programming model. Blue rectangles denote oneDNN objects\, and red lines denote dependencies between objects.,80%,}

## Logical Tensor

*Logical tensor* (@ref dnnl::graph::logical_tensor) describes the metadata of
the input or output tensor, like data type, number of dimensions, size
for each dimension, tensor layout and property. Each logical tensor has a unique
ID and is immutable. Users cannot modify the metadata of a logical tensor
without creating a new one.

## Op

*Op* (@ref dnnl::graph::op) represents an operation as part of a computation
graph. An operation has kind, attribute, and input and output logical tensors.
Operations are added into the graph being constructed. As both operation and
logical tensor contain a unique ID, the graph knows how to connect a producer
operation to a consumer operation through a logical tensor.

## Graph

*Graph* (@ref dnnl::graph::graph) contains a set of operations. A graph object
is associated to a specific engine kind (@ref dnnl::engine::kind). Users can add
multiple operations (@ref dnnl::graph::graph::add_op) and their input logical
tensors and output logical tensors to a graph. After finished adding operations,
users need call finalization API (@ref dnnl::graph::graph::finalize) to indicate
that the graph is ready for partitioning. By calling partitioning API
(@ref dnnl::graph::get_partitions), users will get filtered partitions from the
graph.

## Partition

*Partition* (@ref dnnl::graph::partition) represents a collection of operations
identified by oneDNN Graph API implementation as the basic unit for compilation
and execution. A partition is a connected subgraph within a graph. oneDNN Graph
API implementation analyzes a graph and returns a vector of partitions. The
returned partitions must not form a dependence cycle.

A partition must be compiled (@ref dnnl::graph::partition::compile) before
execution. The compilation lowers down the compute logic to hardware ISA level
and generates binary code. The generated code is specialized for the input and
output logical tensors and engine (@ref dnnl::engine).

The output logical tensor can contain unknown dimensions in which case the
compilation will deduce the output shapes according to input shapes. The output
logical tensor can also have `any` layout type
(@ref dnnl::graph::logical_tensor::layout_type::any). It means that the
compilation will choose the optimal layout for the output tensor. This optimal
layout will be represented as an opaque layout ID saved in the output logical
tensor.

The inputs and outputs of a partition are also called *ports*. The ports record
the logical tensor information passed during graph building stage. Users can
call `get_input_ports` (@ref dnnl::graph::partition::get_in_ports) and
`get_out_ports` (@ref dnnl::graph::partition::get_out_ports) to query them.
These ports can be used to track the producer and consumer relationship between
partitions (through logical tensor ID). The input logical tensors and output
logical tensors must match IDs with ports.

## Compiled Partition

*Compiled partition* (@ref dnnl::graph::compiled_partition) represents the
generated code specialized for target hardware and tensor metadata passed by
compilation API. To execute (@ref dnnl::graph::compiled_partition::execute) a
compiled partition, users must pass input and output tensors and a stream
(@ref dnnl::stream). Input tensors must bind input data buffers to logical
tensors.

Users can query output logical tensor
(@ref dnnl::graph::compiled_partition::query_logical_tensor) from a compiled
partition to know the output layout and memory size
(@ref dnnl::graph::logical_tensor::get_size) when they specify output logical
tensor with `any` layout type during compilation.

## Tensor

*Tensor* (@ref dnnl::graph::tensor) is an abstraction for multidimensional input
and output data needed in the execution of a compiled partition. A tensor
contains a logical tensor, an engine (@ref dnnl::engine) and a data handle.
Users are responsible for managing the tensor’s lifecycle, e.g. free the
resource allocated, when it is not used anymore.
