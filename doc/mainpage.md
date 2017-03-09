A Performance Library for Deep Learning {#mainpage}
================

Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN) is an
open source performance library for Deep Learning (DL) applications intended
for acceleration of DL frameworks on Intel(R) architecture. Intel MKL-DNN
includes highly vectorized and threaded building blocks for implementation of
convolutional neural networks (CNN) with C and C++ interfaces. This
project is created to help DL community innovate on Intel(R) processors.

The library supports the most commonly used primitives necessary to accelerate
bleeding edge image recognition topologies, including Cifar*, AlexNet*, VGG*,
GoogleNet* and ResNet*. The primitives include convolution, inner product,
pooling, normalization, and activation primitives with support for inference
operation. This release includes the following classes of functions:

* Convolution: direct batched convolution,

* Inner Product,

* Pooling: maximum, average,

* Normalization: local response normalization (LRN) across channels and within
                 channel, batch normalization,

* Activation: rectified linear unit neuron activation (ReLU), softmax,

* Data manipulation: reorder (multi-dimensional transposition/conversion), sum,
                     concat, view.

Intel MKL DNN primitives implement a plain C/C++ application programming
interface (API) that can be used in the existing C/C++ DNN frameworks, as well
as in custom DNN applications.

## Programming Model

In Intel MKL-DNN, memory is modeled as a primitive similar to an operation
primitive.  This feature enables reconstructing the graph of computations
at run time.

### Basic Terminology

Intel MKL-DNN operates on the following main objects:

* **Primitive** - any operation: convolution, data format reorder, and even
  memory. Primitives can have other primitives as inputs, but can have only
  memory primitives as outputs.

* **Engine** - an execution device. Currently the only supported engine is CPU.
  Every primitive is mapped to a specific engine.

* **Stream** - an execution context: you submit primitives to a stream and
  wait for their completion. Primitives submitted to a stream may have
  different engines. Stream also tracks dependencies between the primitives.

A typical workflow is the following: create a set of primitives to run,
push them to a stream all at once or one-by-one, and wait for the completion.

### Creating Primitives

In Intel MKL-DNN creation of primitives goes through following levels of
abstraction:

* **Operation/memory descriptor** - a high-level description with logical
  parameters of an operation/memory. It is a lightweight structure, which does
  not allocate any physical memory or computation resources.

* **Primitive descriptor** - a complete description of a primitive that contains
  an operation descriptor, descriptors of primitive inputs and outputs, and the
  target engine. Permits future API extensions to enable querying the descriptor
  for estimated performance, memory consumptions, and so on. A primitive
  descriptor is also a lightweight structure.

* **Primitive** - a specific instance of a primitive created using the
  corresponding primitive descriptor. A primitive structure contains pointers to
  input primitives and output memory. Creation of a primitive is a potentially
  expensive operation because when a primitive is created, Intel MKL-DNN
  allocates resources that are needed to execute the primitive.

To create a memory primitive, follow these steps:

1. Create a memory descriptor. It contains the dimensions, precision, and
   format of data layout in memory. The data layout can be either user specified
   or set to `any`. `any` format is used to enable the operation primitives
   (convolution and inner product) choose the memory format for optimal
   performance.
2. Create a memory primitive descriptor. It contains the memory
   descriptor and the
   target engine.
3. Create a memory primitive. This requires allocating a memory buffer and
   attaching the data handle to the memory primitive descriptor.
   Note, in C++ api for creating an output memory primitive, user
   doesn't need to allocate buffer, unless the output is needed in a
   user format.

To create an operation primitive, follow these steps:

1. Create a logical description of the operation. For example: the description
   of a convolution operation contains parameters such as sizes, strides,
   propagation type, and so on. It will also contain the memory descriptors of
   input and output.
2. Create a primitive descriptor by attaching the target engine to the logical
   description.
3. Create an instance of a primitive and specify the input and output
   primitives.

### Performance Considerations:

*  Convolution and innerproduct primitives when created with unspecified memory
   format `any` for input and/or output, choose the memory format.
   This is choice is based on different circumstances (like hardware,
   convolutional parameters etc..).
*  Operation primitives (e.g. ReLU, LRN, pooling) following convolution or
   innerproduct, should be given input in the same memory format as decided by
   convolution or inner-product. Reorder is a potentially expensive
   operation, so it should be avoided unless needed for performance in
   convolution, innerproduct or output specifications by user.
*  Pooling, concat and sum can be created with output memory format `any`.
*  An operation primitive (typically operations like  pooling, LRN or softmax )
   may need workspace memory for storing results of intermmediate operations
   which are helpful in backward propagation.

### Some operational details:

*  A reorder primitive may need to be created for converting the data from user
   format to a format preferred by convolution or innerproduct.
*  All operations should be queried for requirement of workspace memory.
   If workspace is needed, it should only be created during the forward
   propagation and then shared with corresponding primitive on
   backward propagation.
*  A primitive descriptor from forward propagation must be provided while
   creating corresponding primitive descriptor for backward propagation. This
   allows backward operation to know what exact implementation is chosen for
   the primitive on forward propagation. This in turn helps backward operation
   to decode the workspace memory correctly.
*  User should always check the correspondance between current data format and
   the format that is required by a primitive. For instance, forward convolution
   and backward convolution with respect to source might choose different memory
   formats for weights (if created with `any`). Create a reorder for weights in
   such case. Similarly a reorder might be required for source data between
   forward convolution and backward convolution with respect to weights.

   **Note:** Please refer to extended examples for illustration of the above
   details.

### Auxiliary Types

* **Primitive_at** - a structure that contains a primitive and an index. This
  structure specifies which output of the primitive to use as an input for
  another primitive. For a memory primitive the index is always `0`
  because it doesn't have a output.

### C++ API Example

The repository contains an example of how to build a neural network topology
block that consists of forward convolution and ReLU.

Subtleties to note in this example:

1. How is a reorder primitive created?
2. How is output from convolution passed as input to ReLU?.

Let's go through it step by step:

1. Initialize a CPU engine. The last parameter stands for the index of the
   engine.
~~~cpp
using namespace mkldnn;
auto cpu_engine = engine(engine::cpu, 0);
~~~

2. Create a vector of primitives that represents the net.
~~~cpp
std::vector<primitive> net;
~~~

3. Allocate input data and create a tensor structure that describes it.
~~~cpp
std::vector<float> src(2 * 3 * 227 * 227);
memory::dims conv_src_tz = {2, 3, 227, 227};
/* similarly specify tensor structure for output, weights and bias */
~~~

4. Create two memory descriptors: one for data in a user format, and one for
   the convolution input. Choose `nchw` (minibatch-channels-height-width)
   format for user data and the wildcard `any` for the convolution data format.
  `any` enables the convolution primitive to choose the data format
   that is most suitable for its input parameters (convolution kernel
   sizes, strides, padding, and so on). If the resulting format is different
   from `nchw`, user data needs to be transformed to the format required for
   the convolution.
~~~cpp
auto user_src_md = memory::desc({conv_src_tz},
    memory::data_type::f32, memory::format::nchw);
auto conv_src_md = memory::desc({conv_src_tz},
    memory::data_type::f32, memory::format::any);
/* similarly create conv_weights_md and conv_dst_md in format::any */
~~~

5. Create a convolution descriptor by specifying the algorithm, propagation
   kind, shapes of input, weights, bias, and output, and convolution strides,
   padding, and padding kind.
~~~cpp
auto conv_desc = convolution_forward::desc(
    prop_kind::forward, algorithm::convolution_direct,
    conv_src_md, /* format::any used here to let convolution choose a format */
    conv_weights_md, conv_bias_md, conv_dst_md,
    {1, 1}, {0, 0}, {0, 0}, padding_kind::zero);
~~~

6. Create a descriptor of the convolution primitive. Once created, this
   descriptor has specific formats instead of any wildcard formats specified
   in the convolution descriptor.
~~~cpp
auto conv_pd = convolution_forward::primitive_desc(conv_desc, cpu_engine);
~~~

7. Create a memory primitive that contains user data and check whether the user
   data format differs from the format that the convolution requires. In the
   case of differences, create a reorder primitive that transforms the user data
   to the convolution format and add it to the net.
~~~cpp
auto user_src_memory_descriptor
    = memory::primitive_desc(user_src_md, engine);

auto user_src_memory = memory(user_src_memory_descriptor, src.data());

/* Check whether a reorder is needed  */
auto conv_src_memory = user_src_memory;
if (memory::primitive_desc(conv_pd.src_primitive_desc())
        != user_src_memory_descriptor) {
    /* Yes, it is needed */

    /* Convolution primitive descriptor contains the descriptor of a memory
     * primitive it requires as input. Because a pointer to the allocated
     * memory is not specified, Intel MKL-DNN allocates the memory. */
    conv_src_memory = memory(conv_pd.src_primitive_desc());

    /* create a reorder between data, make it an input for the convolution */
    conv_reorder_src = reorder(user_src_memory, conv_src_memory)

    /* put the reorder in the net */
    net.push_back(conv_reorder_src);
}
~~~

7. Create a memory primitive for output.
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
    innerproduct is encountered) input data format same as format chosen by 
   convolution.
~~~cpp
auto relu_src_md = conv_pd.dst_primitive_desc().desc();

auto relu_desc = relu_forward::desc(prop_kind::forward, relu_src_md,
        negative_slope);
auto relu_dst_memory = memory(relu_pd.dst_primitive_desc());
~~~

11. **Note:** pass convolution primitive as input to relu. This let's the stream
establish dependencies between primitives.
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

Following examples provide more details about using the api. All the examples
use the topology: Convolution, ReLU, LRN and pooling.

* simple_net.c : uses C api. Demonstrates creation of forward primitives.
* simple_net.cpp : uses C++ api.
* simple_training.c: Demonstrates creation of full training net (forward and
   backward primitives) using C api.
* simple_training_net.cpp: uses C++ api.

@subpage legal_information

