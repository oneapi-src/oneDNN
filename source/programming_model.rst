.. include:: replacements.inc.rst

=================
Programming Model
=================

oneDNN Graph main concepts are graph and partition. User constructs a graph with
operations and logical tensors, passes it to oneDNN Graph implementation to get
partitions. OP is an abstraction of compute logic, and it accepts logical
tensors as its input or output data. OP’s kind, attribute, and its input and
output logical tensor defines its semantics. Logical tensor describes the meta
data of the input or output data, like data type, shape, and layout. A logical
tensor must be produced by a single OP and may be consumed by multiple OPs.
oneDNN Graph defines an OP set. oneDNN Graph implementation may support a subset
of the OP set.

Graph is a set of OPs and logical tensors. The order of OP being added to the
graph is considered as the order of OP being executed. Partition is a portion
of the graph, which consists of a set of OP and input and output logical tensors
of each OP. If an input logical tensor is produced by an OP outside its
partition, it becomes partition’s input logical tensor. If an output logical
tensor is used by an OP outside its partition, it becomes partition’s output
logical tensor. oneDNN Graph implementation analyzes a  graph and returns a
number of partitions. The partitions returned don’t contain unsupported OP.

Users may add a wild-card OP to a graph. The wild-card OP represents any compute
logic and its input and output logical tensors contribute to the graph building.
The wild-card OP should not be in any partition. It enables the oneDNN Graph
implementation to analyze a full graph, for example, whether a partition forms
a circular dependence with OPs outside the partition.

A partition needs to be compiled before being used for execution. The
compilation typically lowers down the compute logic to hardware ISA level and
generates binary code. The generated code is specialized for the input data’s
metadata. Typically at the compilation time, users may know more information
like tensor shape. User needs to create parameter logical tensors, corresponding
to partition’s input and output logical tensors, and pass them along with the
compilation API. The oneDNN Graph implementation may decide to use a target
specific layout for the output tensor, and the user may query the logical output
tensor of the compiled partition to know the output tensor’s size.

A compiled partition represents the generated code specialized for target
hardware and meta data described by parameter logical tensors. User needs to
cache the compiled partition to amortize the compilation cost among many
iterations. If parameter tensors’ metadata is identical, a compiled partition
generated in previous iterations may be reused.  Alternatively, implementations
may reduce the partition compilation cost by caching the compiled partition
internally. This optimization falls outside of the scope of this specification.

To execute a compiled partition, users need to attach input data buffers to
logical tensors to create input tensors. Users may prepare the output data
buffers with the queried output tensor sizes and create output tensors. The
execution API takes a compiled partition,  input tensors, and return output
tensors with the modified data.  The partition execution may need to allocate
scratch pad, and it may call a user provided allocator to get the memory.

Engine represents a target device and context in the system. It needs to be
passed as a parameter for partition compilation. Stream is an abstraction for
hardware execution resources of a target device. It is required to execute
a compiled partition.

.. image:: resources/programming_concepts.png

The diagram above describes the key programming concepts, and how they interact
with each other. The arrow indicates the destination object contains or uses the
source object. For example op contains logical tensor, and compiled partition
uses partition.


--------------
Logical Tensor
--------------

Logical tensor describes the meta data of the input or output tensor, like
element data type, number of dimensions, size for each dimension (shape),
layout, and the total size of data.

The “STRIDED” layout_type means that the layout is determined by the strided
field, and “OPAQUE” means that the layout is a private layout decided by oneDNN
Graph implementation.  User may specify layout_type as “ANY” when a parameter
logical tensor is passed in the partition compilation stage to allow oneDNN
Graph implementation to decide the layout for the compiled partition.

.. doxygenclass:: llga::api::logical_tensor
   :project: oneDNN Graph Library
   :members:

--
OP
--

OP is an abstraction of compute logic for deep neural network operation. OP’s
kind, attribute, and its input and output logical tensor defines its semantics.

Conv op contains format attributes for both activation and weight tensor, to
indicate the semantics of each dimension of tensors. For example, the 2D conv
may specify the dimension order is either “NHWC” or “NCHW”. oneDNN Graph uses
one letter “X” to generalize all the spatial dimensions so “NXC” or “NCX” are
used for the last example.

.. doxygenclass:: llga::api::op
   :project: oneDNN Graph Library
   :members:

-----
Graph
-----

Graph is a set of OPs and logical tensors. Add_op() adds an OP and its logical
tensors to a graph.  oneDNN Graph implementation accumulates the OPs and logical
tensors and constructs and validates the graph as internal state. At the end of
graph construction, users may call get_partitions() which returns a set of
partitions. After get_partition(), users shall not add ops to the graph.

A same logical tensor may appear more than twice in add_op() call, since it is
passed with the producer OP and consumer OPs. oneDNN Graph validates logical
tensors with the same id should be identical at the graph construction time.

The order of OP being added to graph is considered as the order of OP being
executed. The returned partitions should not contain OP not supported by the
oneDNN Graph API implementation. The partitions should not contain wild-card OP.
Each partition should not form cyclic dependence with other partitions and OPs
outside partitions.

The logical tensor passed at the graph construction stage might contain
incomplete information, for example, dimension and shape information are
spatially known. Complete information is not required but helps the oneDNN
Graph to form better partition decisions.

.. doxygenclass:: llga::api::graph
   :project: oneDNN Graph Library
   :members:

---------
Partition
---------

Partition represents a collection of OPS identified by oneDNN Graph
implementation as the basic unit for compilation and execution. It contains
a list of oneDNN Graph OPs and their input and output logical tensors. If an
input logical tensor is produced by an OP outside its partition, it becomes
partition’s input logical tensor. If an output logical tensor is used by an OP
outside its partition, it becomes partition’s output logical tensor. User can
not directly access parition’s logical tensors

Partition can infer the output logical tensor’s shape according to the input
logical tensor shape. Users may create input and output logical tensors and pass
them as parameters in the shape inference API. The shape inference API modifies
the parameter logical tensor, and users may get the shape information from the
output logical tensor.

.. note::
   Users manage the lifecycle of oneDNN Graph partitions. After a partition is
   created, users must and should keep it alive before it is compiled to be
   compiled_partition.

Compile() generates hardware ISA level binary code specialized for the metadata
of input and output like tensor shape. Users may also want to build a partition
without shape information so that it won't cause a significant delay when an
unknown shape is fed after the model is deployed. The API supports partition
compilation with or without the tensor shape information, but dynamic shape
compilation is implementation dependent.

The parameter logical tensors must match the id of the logical tensors of the
graph partition captured in the graph construction. Users must specify
“STRIDED”, “ANY”, or “OPAQUE” as the layout_type for the parameter logical
tensors. When users specifies “ANY”, oneDNN Graph implementation decides the
best performant layout for the compiled partition. If it is “STRIDED”, it must
use the public data layout described by the logical tensor. For “OPAQUE”, the
parameter logical tensor contains a target specific layout, and oneDNN Graph may
or may not use the target specific layout.

User must pass input/output data type using logical_tensor, the backend shall
check whether the data type is supported.

User constructs inputs/outputs list based on the order in the modified framework
graph. The backend shall follow this order to get inputs and return outputs.

.. doxygenclass:: llga::api::partition
   :project: oneDNN Graph Library
   :members:

------
Tensor
------

Tensor is an abstraction for multidimensional input and output data needed in
the execution of a compiled partition. A tensor contains a logical tensor and
a data buffer.

Framework integration code is responsible for managing the tensor’s lifecycle,
e.g. free the resource allocated, when it is not used any more. The oneDNN Graph
compiled partition execution may allocate a new tensor with various life cycle
scope, which may need framework’s help to free the tensor at the end of the life
cycle, refer the APIs in allocator.

.. doxygenclass:: llga::api::tensor
   :project: oneDNN Graph Library
   :members:

------------------
Compiled Partition
------------------

A compiled partition represents the generated code specialized for target
hardware and meta data described by parameter logical tensors. Compiled
partition contains a partition and a handle representing the target specific
compiled object.

Users may query the logical tensor from a compiled partition to get the target
specific layout. Users may pass the layout information for next partition
compilation so that it can be optimized expecting a specific input layout.
Users may query the memory size information for the output logical tensor, so
that it can prepare the memory buffer for the output tensor.

Framework passes the parameter tensors and compiled partition to execution API
for execution. Execution API binds the tensor buffers with the parition’s
internal input/output logical tensor and submits it to device runtime for
execution. The parameter logical tensors must be in the same order when they are
passed in the compilation API, and their ids must match with the compiled
partition’s internal logical tensors. The layout type must be “strided” or
“opaque”, can’t be “ANY”.

If users place a tensor with data buffer pointer in outputs, the backend shall
use the data buffer provided by users.

Users may use prepack() to convert the parameter tensor with public layout to
the target specific layout expected by the compiled partition. A common
optimization in deep learning inference is that users may prepack the weight in
the target specific layout required by the compiled partition and cache the
reordered weight for late use.

.. doxygenclass:: llga::api::compiled_partition
   :project: oneDNN Graph Library
   :members:

------
Engine
------

Dnnl::graph::engine represents a device and its context. Compiled partitions are
associated with engines. A compiled partition should only access the tensor
which is associated to the same device and context, no matter the tensor is
produced by a compiled partition or created directly by the user.

Engine contains a device id, device name, and device handle.  Since the engine
serves as the context for oneDNN Graph implementation to store persistent
information, like the compiled partition and its associated persistent memory
cache created on the device, the device id ensures that there is a unique engine
being created for each device.  From the device kind, the engine knows how to
generate code for the target device and what kind of device object to be
expected. The device handle passed from framework allows oneDNN Graph
implementation to work on the device specified by framework.

User program may access the device directly and interoperates with oneDNN Graph
to perform a task on the device. Typically user program manages the device,
which creates the device handle and uses that to create a oneDNN Graph engine.
User program can generate a tensor on a device and pass it to a compiled
partition associated with that engine.

.. doxygenclass:: llga::api::engine
   :project: oneDNN Graph Library
   :members:

------
Stream
------

Stream is the logical abstraction for execution units. It is created on top of
oneDNN Graph engine and typically contains an opencl queue. Each oneDNN Graph
engine may have multiple streams. The compiled partition is submitted to stream
for execution.

.. doxygenclass:: llga::api::stream
   :project: oneDNN Graph Library
   :members:

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
