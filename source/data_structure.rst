==============
Data structure
==============

--
OP
--

LLGA OP describes pure logical description of deep learning operators. It contains inputs, and outputs and attributes. The inputs and outputs tensors are Values that contain logical tensors without concrete tensor data. At the graph partition time, it may not have dimension and shape information until the graph compilation time.

LLGA OP is created when the corresponding framework op is visited, once the information is extracted and converted to LLGA OP, it is then passed over to LLGA backend.

The integration layer manages the OP’s lifecycle, e.g., it frees LLGA OP after it is passed to backend. The LLGA backend should not assume that LLGA OP is alive after it receives a LLGA OP.

.. literalinclude:: code_snippets/op.cpp
    :language: cpp

* ``Op_kind`` - The OP kind specifies which computation is represented by the OP, such as conv2d and relu.
* ``inputs`` - Describes the list of input tensors. Its data type is logical tensor.
* ``outputs`` - Describes the list of output tensors. Its data type is logical tensor.
* ``id`` - Each OP has a unique id to identify itself.
* ``attributes`` - The attributes contain constant information known to the operation at the time of operation creation. Like the K in top K operation, or stride and padding information in conv.

--------------
Logical Tensor
--------------

For logical tensor, the value contains element data type, number of dimensions, size for each dimension (shape). The dimension and shape information may be unknown at the graph partitioning time. LLGA uses DNNL memory descriptor for logical_tensor type.

.. literalinclude:: code_snippets/logical_tensor.cpp
    :language: cpp

------
Policy
------

The policy allows framework to control the size of partitioning.

.. literalinclude:: code_snippets/policy.cpp
    :language: cpp


Discussions: The other potential policy is to pass the max OP number contained within a partition.

---------
Partition
---------

LLGA partition represents a collection of LLGA OPS identified by LLGA backend as the basic unit for compilation and execution. It contains a list of LLGA OPS, input and output values, and a mapping between partition’s input/output to internal LLGA OPs’ input/output.

LLGA integration layer converts the framework op and uses partition API to LLGA backend, which builds its own graph and decides the partition. The LLGA backend manages the lifecycle of LLGA partition. It creates an LLGA partition and should keep it alive before it is compiled to be  compiled_partition.

.. literalinclude:: code_snippets/partition.cpp
    :language: cpp

------------------
Compiled Partition
------------------

LLGA compiled partition represents the compiled object that LLGA backend built for future execution. It has a similar structure as a partition . Other than a list of LLGA OPs, it contains a handle to the internal representation of compiled object.

.. literalinclude:: code_snippets/compiled_partition.cpp
    :language: cpp

------
Tensor
------

Tensor is the core data structure which represents the multidimensional data read and write during the execution of the compiled partition. LLGA compiled partition accepts two types of tensors: plain format tensor and opaque tensor.  Both tensor describes the logical information about the tensor, like element data type, number of dimension, size for each dimension. Plain format tensor describes the physical layout but opaque tensor doesn’t. The plain format tensor has strided representations, which has size and stride for each dimension, and for 2D conv the dimension order is either ``NHWC`` or ``NCHW``.  The plain format tensor also contains a point to the memory buffer.

The opaque tensor doesn’t describe the layout, instead it contains a handle to LLGA backend’s private tensor representation. The opaque tensor allows the tensor data remains private data format between two LLGA partitions, so there is no need for the tensor data to convert back and forth.  which points to platform dependent tensor implementation and doesn't tell the specific data format and location. For the framework integration which does not accept opaque tensor design temporarily, the LLGA may accept the public format tensor and enforce the LLGA backend to accept the public format.

.. literalinclude:: code_snippets/tensor.cpp
    :language: cpp

LLGA interface supports conversion of opaque tensor and public format tensor. **Framework integration code is responsible for managing the tensor’s lifecycle, e.g. free the resource allocated, when it is not used any more. The LLGA compiled partition execution may allocate a new tensor with various life cycle scope, which may need framework’s help to free the tensor at the end of the life cycle.**

------------
Device Model
------------

LLGA assumes that framework has the device model and manages the life cycle of hardware resources while LLGA device model is only a wrapper of the framework device model. The basic requirement of LLGA on framework device model is that it has a ``device`` representing a HW device containing compute and memory resources and a ``stream`` that schedules computation and memory accesses.

LLGA can wrap two types of devices:
1) DPCPP device that is compliant with DPCPP runtime and
2) opaque device that relies on its own device runtime.
The former has the benefit that it shares the same DPCPP runtime with other DPCPP devices so that synchronization can happen without CPU host. But it requires the device vendor to support DPCPP via L0 device model. The latter allows more flexibility of plugging acceleration device model but synchronization with other DPCPP devices always requires CPU host.

------
Engine
------

Engine represents hardware resources including the processing units and memory reserved to support the computation dispatched to the Engine.  It typically contains a device and its associated context.  The LLGA backend may hook persistent information to the engine, like the compiled partition and persistent memory cache used for the partition.

LLGA engine supports two scenarios of how a device is integrated into the framework: 1) framework manages the device with its device model and owns its handle; 2) LLGA backend manages the device and owns its handle. For the former case, framework creates the device handle and passes it tcco the LLGA backend via LLGA engine which acts as a wrapper around the handle. The latter case is suitable for the situation in which the HW device vendor does not integrate the device into the device model of the framework and the framework relies solely on LLGA to interact with the underlying device. In this case, the framework creates the device handle via the LLGA engine by passing the corresponding ``engine_kind`` and ``device_id``.

HW device vendors have the option to support the device with DPC++ device model. In this case, the flag ``is_dpcpp`` is true indicating the corresponding device handle can be casted to a pointer to sycl::device. This gives the framework a chance to better schedule tasks from LLGA backend and other sources within DPC++ runtime without the need of explicit synchronization via host processors. If the flag ``is_dpcpp`` is false, the device handle is opaque to the framework and only interpretable by the LLGA backend.

.. literalinclude:: code_snippets/engine.cpp
    :language: cpp

------
Stream
------

Stream is the logical abstraction for processing units. It is created on top of LLGA engine and typically contains an opencl queue or a thread pool. One LLGA engine may have multiple streams. The compiled partition is submitted to stream for execution.

Similar to DNNL stream, LLGA stream can attach a stream attribute so that frameworks can pass additional information like thread pool information to the LLGA backend.

.. literalinclude:: code_snippets/stream.cpp
    :language: cpp

