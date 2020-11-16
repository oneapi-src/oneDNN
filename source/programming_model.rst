.. SPDX-FileCopyrightText: 2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

=================
Programming Model
=================

oneDNN Graph programming model allows users to pass a computation graph and get
partitions. Users then compile partitions, bind tensor data, and execute
compiled partitions. Partitions are decided by oneDNN Graph implementation,
which is the key concept to satisfy the different needs of AI hardware classes
using a unified API.

The programming model assumes that the main usage is to support deep learning
(DL) frameworks or inference engines. DL frameworks have their own
representation for the computation graph. oneDNN Graph API is used to offload or
accelerate graph partitions from a framework graph. In the description below,
“graph” refers to the graph built by oneDNN Graph implementation, and
“framework graph” refers to the graph built by the DL framework.

A deep learning computation graph consists of deep neural network (DNN)
operations. A DNN operation is a function that takes input data and returns
output data. The input and output data are multi-dimensional arrays called
tensors. A DNN operation may consume multiple tensors and produce multiple
tensors. A tensor must be produced by a single operation and may be consumed by
multiple operations.

oneDNN Graph API uses logical tensor, OP, and graph to represent a computation
graph. Logical tensor represents tensor’s metadata, like element data type,
shape, and layout. OP represents an operation on a computation graph. OP has
kind, attribute, and input and output logical tensors. OPs are added to a graph.
Both OP and logical tensor contains a unique ID, so that the graph knows how to
connect a producer OP to a consumer OP through a logical tensor. The graph
constructed is immutable. The sole purpose of the graph is to partition the
graph. Once users get partitions, users should not add OP to the graph. The
order of OPs being added to the graph is considered as the order of OP being
executed.

oneDNN Graph defines operation set. Users should convert their DNN operation
definition to oneDNN Graph operation for graph construction. For operation
outside oneDNN Graph operation set, users may use wild-card OP. The wild-card OP
represents any OP. With its input and output logical tensors, it enables the
oneDNN Graph implementation to receive a full graph and conduct a complete
analysis. Users must pass data type for each logical tensor. If the tensor shape
and layout are known, users must pass them along with the logical tensor.

A partition is a connected subgraph in a graph. oneDNN Graph implementation
analyzes a  graph and returns a number of partitions. The returned partitions
must not form a dependence cycle. For example, a graph contains 3 OPs: A, B, and
C. If C consumes A’s output and produces B’s input, A and B must not belong to
one partition. If C is not added to the graph, the return partition may include
A and B, since C is not visible to oneDNN Graph implementation. In this case, it
is the user’s responsibility to detect the dependence cycle. Once C is added to
the graph as a wild-card OP, oneDNN Graph implementation is responsible to avoid
the dependence cycle and not put A and B to one partition.

oneDNN Graph implementation may not support every OP in the oneDNN Graph
operation set. The returned partitions don’t contain unsupported OP and
wild-card OP.

A partition needs to be compiled before execution. The compilation lowers down
the compute logic to hardware ISA level and generates binary code. The generated
code is specialized for the input and output tensor’s metadata. Users should
pass as much information about tensor layout as possible in order to get the
best performant compiled code. Users may collect profiling information for
tensor shape or conduct online-compilation when executing the framework graph.
Users must specify the layout for each logical tensor during compilation. Users
must create new logical tensors to pass the enriched metadata with the
compilation API. The new logical tensors must have the same IDs as the logical
tensors passed at the graph construction time.

For the output logical tensors, users must either specify a public layout using
size and stride for each tensor dimension or request oneDNN Graph implementation
to decide a target-specific layout. For the input logical tensors, users must
either specify a public layout or using a target-specific layout produced by
predecessor partition compilation.  For the logical tensor with target-specific
layout, it must be produced by a partition and used only by partitions.

A compiled partition represents the generated code specialized for target
hardware and tensor metadata passed with compilation API. Users may cache the
compiled partition to amortize the compilation cost among many iterations. If
tensor metadata is identical, a compiled partition generated in previous
iterations may be reused.  Alternatively, implementations may reduce the
partition compilation cost by caching the compiled partition internally. This
optimization falls outside of the scope of this specification.

To execute a compiled partition, users must pass input and output tensors. Input
tensors must bind input data buffers to logical tensors. Users may query the
compiled partition for output data buffer sizes. If the sizes are known, users
may allocate the output data buffers and bind to output tensors. If the sizes
are unknown, users must provide an allocator for oneDNN Graph implementation to
allocate the output tensor buffer. The execution API takes a compiled partition,
input tensors, and return output tensors with the data buffer updated.

An engine represents a target device and context in the system. It needs to be
passed as a parameter for partition compilation. A stream abstracts hardware
execution resources of a target device. It is required to execute a compiled
partition.

.. image:: resources/programming_concepts.png

The diagram above summarizes the key programming concepts, and how they interact
with each other. The arrow indicates the destination object contains or uses the
source object. For example, OP contains logical tensor, and compiled partition
uses partition.

--------------
Logical Tensor
--------------

*Logical tensor* describes the metadata of the input or output tensor, like
element data type, number of dimensions, size for each dimension, layout.

Besides helping oneDNN Graph implementation to build the graph, Logical tensor
plays a critical role to exchange tensor metadata information between users and
oneDNN Graph implementation. Users pass input tensor shape information and get
the inferred shape for output tensors from a partition. Users pass logical
tensors to compilation API for specifying shape and layout information. Users
also use a special logical tensor to allow oneDNN Graph implementation to decide
the layout for output tensors. After compilation, users can query the compiled
partition for output tensors’ shape, layout, and sizes.

Each logical tensor has an ID. The tensor metadata may be enriched in the
framework graph as it progresses toward final execution. Logical tensor is not
mutable. Users must create a new logical tensor with the same ID to pass any new
additional information to oneDNN Graph implementation.

.. literalinclude:: code_snippets/logical_tensor.hpp
   :language: cpp

--
OP
--

*OP* describes a deep neural network operation. OP contains kind, attribute, and
input and output logical tensor.

Conv op contains format attributes for both activation and weight tensor, to
indicate the semantics of each dimension of tensors. For example, the 2D conv
may specify the dimension order is either ``NHWC`` or ``NCHW``. oneDNN Graph
uses one letter ``X`` to generalize all the spatial dimensions so ``NXC`` or
``NCX`` are used for the last example.

.. literalinclude:: code_snippets/op.hpp
   :language: cpp

-----
Graph
-----

*Graph* contains a set of OPs. ``add_op()`` adds an OP and its logical tensors
to a graph. oneDNN Graph implementation accumulates the OPs and logical tensors
and constructs and validates the graph as internal state. At the end of graph
construction, users may call ``get_partitions()`` which returns a set of
partitions. After ``get_partitions()``, users shall not add ops to the graph.
The graph doesn't hold any meaning to the user after partitioning. Users should
free the graph.

A same logical tensor may appear more than twice in ``add_op()`` call, since it
is passed with the producer OP and consumer OPs. oneDNN Graph validates logical
tensors with the same id should be identical at the graph construction time.

The order of OP being added to the graph is considered as the order of OP being
executed. The returned partitions should not contain OP not supported by the
oneDNN Graph API implementation. The partitions should not contain wild-card OP.
Partitions should not form cyclic dependence within the graph. It is the user's
responsibility to detect any dependence cycle between the partitions and
operations not passing to oneDNN Graph implementation.

The logical tensor passed at the graph construction stage might contain
incomplete information, for example, dimension and shape information are
spatially known. Complete information is not required but helps the oneDNN Graph
to form better partition decisions.

.. literalinclude:: code_snippets/graph.hpp
   :language: cpp

---------
Partition
---------

*Partition* represents a collection of OPS identified by oneDNN Graph
implementation as the basic unit for compilation and execution. It contains a
list of OP IDs.

Partition can infer the output logical tensor’s shape according to the input
logical tensor shape. Users may create input and output logical tensors and pass
them as parameters of the shape inference API. After calling the shape inference
API, users can get the shape information from the output logical tensor.

Partition can be compiled to generate hardware ISA level binary code specialized
for input and output tensors’ metadata. Users must pass as much tensor metadata
as possible to get the best performant compiled code. When users pass partition
shape information, it is implementation-dependent to decide whether to support
the compilation.

Users must create an input logical tensor list and an output logical tensor list
to pass the additional tensor metadata as parameters to the compilation API. The
parameter logical tensors must match the id of the logical tensors of the graph
partition captured in the graph construction.

Users must specify ``strided``, ``any``, or ``opaque`` as the ``layout_type``
for the parameter logical tensors. When users specify ``any`` for a logical
tensor, the tensor must be an output tensor, and oneDNN Graph implementation
decides the best performant layout for the compiled partition. If it is
``strided``, it must use the public data layout described by the logical tensor.
For ``opaque``, the parameter logical tensor contains a target-specific layout,
which must be determined by the compilation of preceding partitions producing
the tensor.

.. literalinclude:: code_snippets/partition.hpp
   :language: cpp

------
Tensor
------

*Tensor* is an abstraction for multidimensional input and output data needed in
the execution of a compiled partition. A tensor contains a logical tensor and a
data buffer.

Framework integration code is responsible for managing the tensor’s lifecycle,
e.g. free the resource allocated, when it is not used anymore.

.. literalinclude:: code_snippets/tensor.hpp
   :language: cpp

------------------
Compiled Partition
------------------

A *compiled partition* represents the generated code specialized for target
hardware and meta data described by parameter logical tensors. Compiled
partition contains a partition and a handle representing the target specific
compiled object.

After the compilation API is invoked, users must query the logical output tensor
of the compiled partition to know the output tensor’s layout id and size. The
layout id is an opaque identifier for the target-specific layout. Users may pass
the layout id for the next partition compilation so that it can be optimized to
expect a specific input layout.  Users may use the size to allocate the memory
buffer of the output tensors for execution.

Framework passes the tensors and compiled partition as parameters to execution
API. The parameter logical tensors must be in the same order when they are
passed in the compilation API, and their ids must match with the compiled
partition’s internal logical tensors. The layout type of each tensor must be
``strided`` or ``opaque``.

If users place a tensor with data buffer pointer in outputs, the backend shall
use the data buffer provided by users.

Users may convert the parameter tensor with public layout to the target specific
layout expected by the compiled partition. A common optimization in deep
learning inference is that users may prepack the weight in the target-specific
layout required by the compiled partition and cache the reordered weight for
late use.

.. literalinclude:: code_snippets/compiled_partition.hpp
   :language: cpp

------
Engine
------

*Engine* represents a device and its context. Compiled partitions are associated
with engines. A compiled partition should only access the tensor which is
associated with the same device and context, no matter the tensor is produced by
a compiled partition or created directly by the user.

Engine contains device kind, and a device id or device handle. From the device
kind, the engine knows how to generate code for the target device and what kind
of device object to be expected. The device id ensures that there is a unique
engine being created for each device. The device handle passed from framework
allows oneDNN Graph implementation to work on the device specified by the
framework.

User programs may access the device directly and interoperate with oneDNN Graph
to perform a task on the device. Typically user programs manage the device,
which create the device handle and use that to create an oneDNN Graph engine.
User programs can generate a tensor on a device and pass it to a compiled
partition associated with that engine.

.. literalinclude:: code_snippets/engine.hpp
   :language: cpp

------
Stream
------

*Stream* is the logical abstraction for execution units. It is created on top of
oneDNN Graph engine. For SYCL device, it contains an openCL queue. oneDNN Graph
engine may have multiple streams. A compiled partition is submitted to a stream
for execution.

.. literalinclude:: code_snippets/stream.hpp
   :language: cpp

-----------------
General API notes
-----------------

There are certain assumptions on how oneDNN Graph objects behave:

* Logical tensor behave similarly to trivial types.
* All other objects behave like shared pointers. Copying is always shallow.

--------------
Error Handling
--------------

The C++ API throws exceptions for error handling.
