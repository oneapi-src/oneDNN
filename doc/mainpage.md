A Performance Library for Deep Learning
================

The Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN) is an
open source performance library for Deep Learning (DL) applications intended
for acceleration of DL frameworks on Intel(R) architecture. Intel MKL-DNN
includes highly vectorized and threaded building blocks for implementation of
convolutional neural networks (CNN) with C and C++ interfaces. This
project is created to help the DL community innovate on the Intel(R) processor family.

The library supports the most commonly used primitives necessary to accelerate
bleeding edge image recognition topologies, including Cifar*, AlexNet*, VGG*,
GoogleNet*, and ResNet*. The primitives include convolution, inner product,
pooling, normalization, and activation with support for inference
operations. The library includes the following classes of functions:

* Convolution
    - direct batched convolution
* Inner Product
* Pooling
    - maximum
    - average
* Normalization
    - local response normalization (LRN) across channels and within channel
    - batch normalization

* Activation
    - rectified linear unit neuron activation (ReLU)
	- softmax

* Data manipulation
    - reorder (multi-dimensional transposition/conversion),
    - sum,
    - concat
	- view

Intel MKL DNN primitives implement a plain C/C++ application programming
interface (API) that can be used in the existing C/C++ DNN frameworks, as well
as in custom DNN applications.

## Programming Model

Intel MKL-DNN models memory as a primitive similar to an operation
primitive.  This allows reconstruction of the graph of computations
at run time.

### Basic Terminology

Intel MKL-DNN operates on the following main objects:

* **Primitive** - any operation, such as convolution, data format reordering, and even
  memory. Primitives can have other primitives as inputs, but can have only
  memory primitives as outputs.

* **Engine** - an execution device. Currently the only supported engine is CPU.
  Every primitive is mapped to a specific engine.

* **Stream** - an execution context. You submit primitives to a stream and
  wait for their completion. Primitives submitted to the same stream can have
  different engines. The stream also tracks dependencies between the primitives.

A typical workflow is to create a set of primitives to run,
push them to a stream all at once or one at a time, and wait for completion.

### Creating Primitives

In Intel MKL-DNN, creating primitives involves three levels of
abstraction:

* **Operation/memory descriptor** - a high-level description with logical
  parameters of an operation or memory. It is a lightweight structure that does
  not allocate any physical memory or computation resources.

* **Primitive descriptor** - a complete description of a primitive that contains
  an operation descriptor, descriptors of primitive inputs and outputs, and the
  target engine. This permits future API extensions to enable querying the descriptor
  for estimated performance, memory consumptions, and so on. A primitive
  descriptor is also a lightweight structure.

* **Primitive** - a specific instance of a primitive created using the
  corresponding primitive descriptor. A primitive structure contains pointers to
  input primitives and output memory. Creating a primitive is a potentially
  expensive operation because when a primitive is created, Intel MKL-DNN
  allocates the necessary resources to execute the primitive.

To create a memory primitive:

1. Create a memory descriptor. The memory descriptor contains the dimensions, precision, and
   format of the data layout in memory. The data layout can be either user-specified
   or set to `any`. The `any` format allows the operation primitives
   (convolution and inner product) to choose the best memory format for optimal
   performance.
2. Create a memory primitive descriptor. The memory primitive descriptor contains the memory
   descriptor and the
   target engine.
3. Create a memory primitive. The memory primitive requires allocating a memory buffer and
   attaching the data handle to the memory primitive descriptor.
   **Note:** in the C++ API for creating an output memory primitive, you
   do not need to allocate buffer unless the output is needed in a
   user-defined format.

To create an operation primitive:

1. Create a logical description of the operation. For example, the description
   of a convolution operation contains parameters such as sizes, strides, and
   propagation type. It also contains the input and outpumemory descriptors.
2. Create a primitive descriptor by attaching the target engine to the logical
   description.
3. Create an instance of the primitive and specify the input and output
   primitives.

### Performance Considerations

*  Convolution and inner product primitives choose the memory format when you create them with the unspecified memory
   format `any` for input or output.
   The memory format chosen is based on different circumstances such as hardware and
   convolutional parameters.
*  Operation primitives (such as ReLU, LRN, or pooling) following convolution or
   inner product, should have input in the same memory format as the
   convolution or inner-product. Reordering can be an expensive
   operation, so you should avoid it unless it is necessary for performance in
   convolution, inner product, or output specifications.
*  Pooling, concat and sum can be created with the output memory format `any`.
*  An operation primitive (typically operations such as pooling, LRN, or softmax)
   might need workspace memory for storing results of intermediate operations
   that help with backward propagation.

### Miscellaneous Operational Details

*  You might need to create a reorder primitive to convert the data from a user
   format to the format preferred by convolution or inner product.
*  All operations should be queried for workspace memory requirements.
   If workspace is needed, it should only be created during the forward
   propagation and then shared with the corresponding primitive on
   backward propagation.
*  A primitive descriptor from forward propagation must be provided while
   creating corresponding primitive descriptor for backward propagation. This
   tells the backward operation what exact implementation is chosen for
   the primitive on forward propagation. This in turn helps the backward operation
   to decode the workspace memory correctly.
*  You should always check the correspondance between current data format and
   the format that is required by a primitive. For instance, forward convolution
   and backward convolution with respect to source might choose different memory
   formats for weights (if created with `any`). In this case, you should create a reorder primitive for weights. 
   Similarly, a reorder primitive might be required for a source data between
   forward convolution and backward convolution with respect to weights.

   **Note:** Please refer to extended examples to illustrate these details.

### Auxiliary Types

* **Primitive_at** - a structure that contains a primitive and an index. This
  structure specifies which output of the primitive to use as an input for
  another primitive. For a memory primitive the index is always `0`
  because it does not have a output.

## Example

This C++ API example demonstrates how to build a neural network topology
block that consists of forward convolution and ReLU.

In this example note:

* how a reorder primitive is created
* how output from convolution passed as input to ReLU

The steps in the example are:

1. Initialize a CPU engine. The last parameter in the engine() call represents the index of the
   engine.
~~~cpp
using namespace mkldnn;
auto cpu_engine = engine(engine::cpu, 0);
~~~

2. Create a vector of primitives that represents the net.
~~~cpp
std::vector<primitive> net;
~~~

3. Allocate input data and create a tensor structure that describes the data.
~~~cpp
std::vector<float> src(2 * 3 * 227 * 227);
memory::dims conv_src_tz = {2, 3, 227, 227};
/* similarly specify tensor structure for output, weights and bias */
~~~

4. Create two memory descriptors: one for data in a user format, and one for
   the convolution input. Choose `nchw` (minibatch-channels-height-width)
   format for user data and `any` for the convolution data format.
   The `any` format allows the convolution primitive to choose the data format
   that is most suitable for its input parameters (convolution kernel
   sizes, strides, padding, and so on). If the resulting format is different
   from `nchw`, the user data must be transformed to the format required for
   the convolution.
~~~cpp
auto user_src_md = memory::desc({conv_src_tz},
    memory::data_type::f32, memory::format::nchw);
auto conv_src_md = memory::desc({conv_src_tz},
    memory::data_type::f32, memory::format::any);
/* similarly create conv_weights_md and conv_dst_md in format::any */
~~~

5. Create a convolution descriptor by specifying the algorithm, propagation
   kind, shapes of input, weights, bias, output, convolution strides,
   padding, and kind of padding.
~~~cpp
auto conv_desc = convolution_forward::desc(
    prop_kind::forward, algorithm::convolution_direct,
    conv_src_md, /* format::any used here to allow convolution choose a format */
    conv_weights_md, conv_bias_md, conv_dst_md,
    {1, 1}, {0, 0}, {0, 0}, padding_kind::zero);
~~~

6. Create a descriptor of the convolution primitive. Once created, this
   descriptor has specific formats instead of the `any` format specified
   in the convolution descriptor.
~~~cpp
auto conv_pd = convolution_forward::primitive_desc(conv_desc, cpu_engine);
~~~

7. Create a memory primitive that contains user data and check whether the user
   data format differs from the format that the convolution requires. In
   case it is different, create a reorder primitive that transforms the user data
   to the convolution format and add it to the net.
~~~cpp
auto user_src_memory_descriptor
    = memory::primitive_desc(user_src_md, engine);

auto user_src_memory = memory(user_src_memory_descriptor, src.data());

/* Check whether a reorder is necessary  */
auto conv_src_memory = user_src_memory;
if (memory::primitive_desc(conv_pd.src_primitive_desc())
        != user_src_memory_descriptor) {
    /* Yes, a reorder is necessary */

    /* The convolution primitive descriptor contains the descriptor of a memory
     * primitive it requires as input. Because a pointer to the allocated
     * memory is not specified, Intel MKL-DNN allocates the memory. */
    conv_src_memory = memory(conv_pd.src_primitive_desc());

    /* create a reorder between data, make it an input for the convolution */
    conv_reorder_src = reorder(user_src_memory, conv_src_memory)

    /* put the reorder in the net */
    net.push_back(conv_reorder_src);
}
~~~

8. Create a memory primitive for output.
~~~cpp
auto conv_dst_memory = memory(conv_pd.dst_primitive_desc());
~~~

9. Create a convolution primitive and add it to the net.
~~~cpp
/* Note that the conv_reorder_src primitive
 * is an input dependency for the convolution primitive, which means that the
 * convolution primitive will not be executed before the data is ready. */
auto conv
        = convolution_forward(conv_pd, conv_src_memory, conv_weights_memory,
                              conv_user_bias_memory, conv_dst_memory);
net.push_back(conv);
~~~

10. Create relu primitive. For better performance keep ReLU
   (as well as for other operation primitives until another convolution or 
    inner product is encountered) input data format in the same format as was chosen by 
   convolution.
~~~cpp
auto relu_src_md = conv_pd.dst_primitive_desc().desc();

auto relu_desc = relu_forward::desc(prop_kind::forward, relu_src_md,
        negative_slope);
auto relu_dst_memory = memory(relu_pd.dst_primitive_desc());
~~~

11. Pass the convolution primitive as input to relu. 
    **Note:** this allows the stream to establish dependencies between primitives.
~~~cpp
auto relu = relu_forward(relu_pd, conv, relu_dst_memory);
net.push_back(relu);
~~~

12. Finally, create a stream, submit all the primitives, and wait for
    completion.
~~~cpp
mkldnn::stream(mkldnn::stream::kind::eager).submit(net).wait();
~~~

### Extended Examples

These examples provide more details about using the API. All the examples
use the topology: Convolution, ReLU, LRN, and pooling.

* Creation of forward primitives
    - C: simple_net.c 
    - C++: simple_net.cpp
  
* Creation of full training net (forward and backward primitives)
    - C: simple_training.c
    - C++: simple_training_net.cpp

--------
	
[Legal information](legal_information.md)
