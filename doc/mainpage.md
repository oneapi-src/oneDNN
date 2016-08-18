# Developer Manual {#mainpage}

Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN) is an
open source performance library for Deep Learning (DL) applications intended
for acceleration of DL frameworks on Intel(R) architecture. Intel MKL-DNN
includes highly vectorized and threaded building blocks for implementation of
convolutional neural networks (CNN) with C and C++ interfaces. We created this
project to help DL community innovate on Intel(R) processors.

The library supports the most commonly used primitives necessary to accelerate
bleeding edge image recognition topologies, including AlexNet and VGG. The
primitives include convolution, inner product, pooling, normalization and
activation primitives with support for forward (scoring or inference)
operations. Current release includes the following classes of functions:

* Convolution: direct batched convolution

* Inner Product

* Pooling: maximum

* Normalization: local response normalization across channels (LRN)

* Activation: rectified linear neuron activation (ReLU)

* Data manipulation: reorder (multi-dimensional transposition/conversion)

Intel(R) MKL DNN primitives implement a plain C application programming
interface (API) that can be used in the existing C/C++ DNN frameworks, as well
as in custom DNN applications.

## Programming Model

In Intel MKL-DNN model primitives have other primitives as inputs. Outputs are
memory primitives only. This makes it possible to reconstruct the graph of
computations at the run time.


### Basic terminology

Intel(R) MKL-DNN operates on the following three main objects:

* **Primitive** - any operation: convolution, data format reorder, and even
  memory. Primitives can have other primitives as inputs but may have only
  memory primitives as outputs.

* **Engine** - an execution device. Currently the only supported engine is CPU.
  Every primitive is mapped to a specific engine.

* **Stream** - an execution context: user submits primitives to a stream and
  waits for their completion. Primitives submitted to a stream can have
  different engines. Stream also track dependencies between the primitives.

The typical workflow is the following: create a set of primitives to be run,
push them all at once or one-by-one into a stream and wait for the completion.

### Compute primitives

To create a primitive, one has to first create a logical description of the
operation such as a memory descriptor or a convolution descriptor. Then they
need to create a description of the primitive to perform the operation with all
the details like descriptions of inputs and outputs defined. Finally, they need
to create a primitive specifying other primitives as inputs and outputs. The
following list describes these levels of abstraction in details:

* **Operation/memory descriptor** - a high level description with logical
  parameters of an operation/memory. For instance, for a convolution operation
  the description would contain parameters such as sizes, strides, propagation
  type, etc. A memory description would contain dimensions, precision and
  format of data layout in memory. The memory format may be set to `any`, which
  means that it is not yet defined. This is used to let primitives choose the
  memory format for optimal performance. This structure is lightweight and does
  not allocate any additional resources.

* **Primitive descriptor** - a complete description of a primitive containing
  an operation descriptor, descriptors of primitive inputs and outputs, and the
  target engine. In the future, it would be possible to user query a primitive
  descriptor for estimated performance, memory consumptions and so on. This is
  also a lightweight structure.

* **Primitive** - a specific instance of a primitive produced using a
  corresponding primitive descriptor. It contains pointers to input primitives
  and output memory. Creation of a primitive is a potentially expensive
  operation because when a primitive is created, MKL-DNN allocates resources
  that are needed to execute it.

### Auxiliary types

* **Tensor** - an data description containing the number of dimensions the data
  has and the dimensions themselves.

* **Primitive_at** - a structure containing a primitive and an index. This
  structure specifies which output of the primitive to use as a input for
  another primitive.

### C++ API example

The repository contains an example how to build a block of neural network
topology consisting of convolution, ReLU, LRN and pooling. Let's go through it
step by step.


1. Initialize a CPU engine. The last parameter stands for the index of the
   engine.
```cpp
auto cpu_engine = mkldnn::engine(mkldnn::engine::cpu, 0);
```

2. Create a vector of primitives which will represent the net.
```cpp
std::vector<mkldnn::primitive> net;
```

3. Allocate input data and create a tensor structure that describes it.
```cpp
std::vector<float> src(2 * 3 * 227 * 227);
mkldnn::tensor::dims conv_src_dims = {2, 3, 227, 227};
```

4. Create two memory descriptors: one for data in users' format, and one for
   the convolution input. We use `nchw` (minibatch-channels-height-width)
   format for user data format and wildcard `any` for the convolution data
   format. The wildcard means that convolution primitive will choose the data
   format that is most suitable for its input parameters (convolutional kernel
   sizes, strides, padding and so on). If the resulting format is different
   from `nchw`, user data will have to be transformed to the format expected by
   convolution.
```cpp
auto user_src_md = mkldnn::memory::desc({conv_src_dims},
    mkldnn::memory::precision::f32, mkldnn::memory::format::nchw);
auto conv_src_md = mkldnn::memory::desc({conv_src_dims},
    mkldnn::memory::precision::f32, mkldnn::memory::format::any);
```

5. Create a convolution descriptor by specifying the algorithm, the propagation
   kind, shapes of input, weights, bias and output, convolution strides,
   padding and padding kind.
```cpp
auto conv_desc = mkldnn::convolution::desc(
    mkldnn::prop_kind::forward, mkldnn::convolution::direct,
    conv_src_md, /* format::any used here to let convolution choose a format */
    conv_weights_md, conv_bias_md, conv_dst_md,
    {1, 1}, {0, 0}, mkldnn::padding_kind::zero);
```

6. Create a convolution primitive descriptor. Once created, the descriptor will
   have specific formats in place of any wildcard formats that were specified
   in the convolution descriptor.
```cpp
auto conv_pd = mkldnn::convolution::primitive_desc(conv_desc, cpu_engine);
```

7. Create a memory primitive containing user's data and check whether user data
   format differs from what convolution expects. If yes, create a reorder
   primitive that transforms user data to the convolution format and add it to
   the net.
```cpp
auto user_src_memory_descriptor
    = mkldnn::memory::primitive_desc(user_src_md, engine);

auto user_src_memory = mkldnn::memory(user_src_memory_descriptor, src);

/* Check if we need a reorder */
auto conv_input = user_src_memory;
if (mkldnn::memory::primitive_desc(conv_pd.data.src_primitive_desc)
        != user_src_memory_descriptor) {
    /* Yes, we do */

    /* Convolution primitive descriptor contains the memory primitive
     * descriptor it expects as it's input. Since we don't specify a
     * pointer to allocated memory, the mmory will be allocated by MKL-DNN */
    auto conv_src_memory = mkldnn::memory(conv_pd.data.src_primitive_desc);

    /* create a reorder between data, make it an input for a convolution */
    conv_input = mkldnn::reorder(user_src_memory, conv_src_memory)

    /* put the reorder in the net */
    net.push_back(conv_input);
}
```

8. Create a convolution primitive and add it to the net.
```cpp
/* Note that the conv_input primitive (whether it is a memory or a reorder)
 * is an input dependency for the convolution primitive which means that the
 * convolution primitive won't be executed before the data is ready. */
auto conv = mkldnn::convolution(conv_pd, conv_input, conv_weights_memory,
        conv_user_bias_memory, conv_dst_memory);
net.push_back(conv);
```

9. Finally, create a stream, submit all the primitives, and wait for
    completion.
```cpp
mkldnn::stream().submit(net).wait();
```

@subpage legal_information

