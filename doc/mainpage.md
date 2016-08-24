A Performance Library for Deep Learning {#mainpage}
================

Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN) is an
open source performance library for Deep Learning (DL) applications intended
for acceleration of DL frameworks on Intel(R) architecture. Intel MKL-DNN
includes highly vectorized and threaded building blocks for implementation of
convolutional neural networks (CNN) with C and C++ interfaces. This
project is created to help DL community innovate on Intel(R) processors.

The library supports the most commonly used primitives necessary to accelerate
bleeding edge image recognition topologies, including AlexNet* and VGG*. The 
primitives include convolution, inner product, pooling, normalization, and
activation primitives with support for forward (scoring or inference)
operations. This release includes the following classes of functions:

* Convolution: direct batched convolution

* Inner Product

* Pooling: maximum

* Normalization: local response normalization across channels (LRN)

* Activation: rectified linear unit neuron activation (ReLU)

* Data manipulation: reorder (multi-dimensional transposition/conversion)

Intel MKL DNN primitives implement a plain C application programming
interface (API) that can be used in the existing C/C++ DNN frameworks, as well
as in custom DNN applications.

## Programming Model

While in Intel MKL-DNN model, primitives have other primitives as inputs, outputs are
memory primitives only. This feature enables reconstructing the graph of
computations at run time.


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

To create a primitive, follow these steps:
1. Create a logical description of the operation in a descriptor such as a memory
   descriptor or convolution descriptor.
2. Create a description of the primitive to perform the operation. The description
   defines all the necessary details, such as descriptions of inputs and outputs.
3. Create an instance of a primitive and specify other primitives as inputs
   and outputs.

These steps reflect the following levels of abstraction:

* **Operation/memory descriptor** - a high-level description with logical
  parameters of an operation/memory. For example: the description of a
  convolution operation contains parameters such as sizes, strides, propagation
  type, and so on. A memory description contains dimensions, precision, and
  format of the data layout in memory. The memory format can be set to `any`, which
  means that it is not yet defined. This format is used to enable primitives choose
  the memory format for optimal performance. An operation/memory descriptor is a
  lightweight structure, which does not allocate any additional resources.

* **Primitive descriptor** - a complete description of a primitive that contains
  an operation descriptor, descriptors of primitive inputs and outputs, and the
  target engine. Permits future API extensions to enable querying the descriptor
  for estimated performance, memory consumptions, and so on. A primitive
  descriptor is also a lightweight structure.

* **Primitive** - a specific instance of a primitive created using the
  corresponding primitive descriptor. A primitive structure contains pointers
  to input primitives and output memory. Creation of a primitive is a potentially
  expensive operation because when a primitive is created, Intel MKL-DNN allocates
  resources that are needed to execute the primitive.

### Auxiliary Types

* **Tensor** - a data description that contains the number of dimensions the data
  has and the dimensions themselves.

* **Primitive_at** - a structure that contains a primitive and an index. This
  structure specifies which output of the primitive to use as an input for
  another primitive.

### C++ API Example

The repository contains an example of how to build a neural network topology
block that consists of convolution, ReLU, LRN, and pooling. Let's go through it
step by step:


1. Initialize a CPU engine. The last parameter stands for the index of the
   engine.
~~~cpp
auto cpu_engine = mkldnn::engine(mkldnn::engine::cpu, 0);
~~~

2. Create a vector of primitives that represents the net.
~~~cpp
std::vector<mkldnn::primitive> net;
~~~

3. Allocate input data and create a tensor structure that describes it.
~~~cpp
std::vector<float> src(2 * 3 * 227 * 227);
mkldnn::tensor::dims conv_src_dims = {2, 3, 227, 227};
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
auto user_src_md = mkldnn::memory::desc({conv_src_dims},
    mkldnn::memory::precision::f32, mkldnn::memory::format::nchw);
auto conv_src_md = mkldnn::memory::desc({conv_src_dims},
    mkldnn::memory::precision::f32, mkldnn::memory::format::any);
~~~

5. Create a convolution descriptor by specifying the algorithm, propagation
   kind, shapes of input, weights, bias, and output, and convolution strides,
   padding, and padding kind.
~~~cpp
auto conv_desc = mkldnn::convolution::desc(
    mkldnn::prop_kind::forward, mkldnn::convolution::direct,
    conv_src_md, /* format::any used here to let convolution choose a format */
    conv_weights_md, conv_bias_md, conv_dst_md,
    {1, 1}, {0, 0}, mkldnn::padding_kind::zero);
~~~

6. Create a descriptor of the convolution primitive. Once created, this descriptor
   has specific formats instead of any wildcard formats specified
   in the convolution descriptor.
~~~cpp
auto conv_pd = mkldnn::convolution::primitive_desc(conv_desc, cpu_engine);
~~~

7. Create a memory primitive that contains user data and check whether the user
   data format differs from the format that the convolution requires. In the case
   of differences, create a reorder primitive that transforms the user data to the
   convolution format and add it to the net.
~~~cpp
auto user_src_memory_descriptor
    = mkldnn::memory::primitive_desc(user_src_md, engine);

auto user_src_memory = mkldnn::memory(user_src_memory_descriptor, src);

/* Check whether a reorder is needed  */
auto conv_input = user_src_memory;
if (mkldnn::memory::primitive_desc(conv_pd.data.src_primitive_desc)
        != user_src_memory_descriptor) {
    /* Yes, it is needed */

    /* Convolution primitive descriptor contains the descriptor of a memory
     * primitive it requires as input. Because a pointer to the allocated
     * memory is not specified, Intel MKL-DNN allocates the memory. */
    auto conv_src_memory = mkldnn::memory(conv_pd.data.src_primitive_desc);

    /* create a reorder between data, make it an input for the convolution */
    conv_input = mkldnn::reorder(user_src_memory, conv_src_memory)

    /* put the reorder in the net */
    net.push_back(conv_input);
}
~~~

8. Create a convolution primitive and add it to the net.
~~~cpp
/* Note that the conv_input primitive (whether it be a memory or a reorder)
 * is an input dependency for the convolution primitive, which means that the
 * convolution primitive will not be executed before the data is ready. */
auto conv = mkldnn::convolution(conv_pd, conv_input, conv_weights_memory,
        conv_user_bias_memory, conv_dst_memory);
net.push_back(conv);
~~~

9. Finally, create a stream, submit all the primitives, and wait for
    completion.
~~~cpp
mkldnn::stream().submit(net).wait();
~~~

@subpage legal_information

