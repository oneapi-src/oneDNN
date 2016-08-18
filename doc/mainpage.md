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
    operations. Current release includes the following clasess of functions:

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
memory primitives only. Makes it possible to reconstruct computations graph at
run-time.

### Basic terminology

Intel(R) MKL-DNN operates with the following three main objects:

* **Primitive** - any operation: convolution, data format reorder, and even
    memory (no-op / identity).  Primitives have other primitives as inputs but
    may have only memory primitives as outputs.

* **Engine** - an execution device. Currently the only supported engine is CPU.
    Every primitive is assigned to specific engine.

* **Stream** - an execution context: user submits primitives to a stream and
    waits for their completion.  Since primitive _remembers_ their engine
    stream can handle primitives with different engines. It also can handle
    non-linear execution, because each primitive knows it's inputs (which are
    also primitive, remember?) hence dependency tracking is possible.

The typical workflow is the following: one creates a set of primitives to be
run, push them at once or one-by-one into a stream and wait for completion.

### Primitive construction hierarhy

To create a primitive one has to go through logical description of the
operation (which is called memory descriptor or operation descriptor depending
on what is being created), descripton of the operation with all the specifics
(which is called primitive descriptor hereinafter) and finally the primitive
itself. The following list describes these levels of abstraction in details:

* **Operation/memory descriptor** - high level description with logical sizes
    of an operation/memory. For instance for convolution operation it will
    contain sizes, strides, propagation type, etc. In case of memory it will
    contail the sizes, precision and layout. The layout might be `any`, which
    means the layout is not yet defined. This is used to let primitive choose
    the memory format for optimal performance.

* **Primitive descriptor** - full description of a primitive. It contains an
    operation descriptor, descriptors of inputs, and outputs, engine.
    Internally primitive descriptor is also contain an information on
    particular implementaion which is going to be used. In future this will
    allow user quering the primitive descriptor for certain characteristics,
    like performance, memory consumptions and so on.

* **Primitive** - A specific implementation of operation (or memory). It also
    contains links on input primitives and output memory. In particular it
    means that primitive has everything it needs to be executed.

### Few additional things you have to know about

Before moving forward to C++ example we need to introduce `few more` types:

* **Tensor** - an _array_ description: number of dimentions and the dimensions
    themselves.

* **Primitive_at** - a structure of a primitive and index. This structure
    allows specifing which output of the primitive to use as input for
    subsequent primitive.

### C++ API example

The repository contains an example how to build a block of neural network
topology consisting of convolution, relu, lrn and pooling. Let's go through it
step by step.


1. Initialize CPU engine. The last parameter stands for index of the engine.
```cpp
auto cpu_engine = engine(engine::cpu, 0);
```

2. Define a vector of primitives, which will represent the net.
```cpp
std::vector<primitive> net;
```

3. Define input data and tensor describes it.
```cpp
std::vector<float> src(2 * 3 * 227 * 227);
tensor::dims conv_src_dims = {2, 3, 227, 227};
```

4. Create two memory desciptors: one describes data in users' format (in this
    example `nchw` is used which stands for natural order of the data:
    minibatch-channels-height-width). The second memory descriptor contain a
    wildcard `any`, which means that convolution will define the layout it wants
    to use to achieve the best performance for this particular case (
    convolutional kernel sizes, strides, padding and so on). In case a
    convolution will choose the layout other then users' one, the reorder is
    required from users' data to convolution input.
```cpp
auto user_src_md = memory::desc({conv_src_dims}, memory::precision::f32, memory::format::nchw);
auto conv_src_md = memory::desc({conv_src_dims}, memory::precision::f32, memory::format::any);
```

5. Create a convolution descriptor, by specifing defining an algorithm, a
    propagation kind, input/weights/bias/output shapes, strides, padding and
    padding policy.
```cpp
auto conv_desc = convolution::desc(
    prop_kind::forward, convolution::direct,
    conv_src_md, /* format::any used here to let convolution decide which layout to use */
    conv_weights_md, conv_bias_md, conv_dst_md,
    {1, 1}, {0, 0}, padding_kind::zero);
```

6. Create a convolution primitive descriptor. Once created all semi-defined
    structures like a layout for the source are defined now.
```cpp
auto conv_pd = convolution::primitive_desc(conv_desc, cpu_engine);
```

7. Create a memory primitive which represents user's data. Also check where
    users' data format differs from what convolution expects. In case memory
    formats differ create a reorder between those two and put it to the net.
```cpp
auto user_src_memory_descriptor = memory::primitive_desc(user_src_md, engine);

auto user_src_memory = memory(user_src_memory_descriptor, src);
auto conv_input = user_src_memory; /* let's hope we won't need a reorder here */

if (memory::primitive_desc(conv_pd.data.src_primitive_desc) != user_src_memory_descriptor) {
    /* unfortunately, we need */

    /* convolution primitive descriptor contains memory primitive
     * descriptor it expects as it's input. since we don't put a
     * pointer here, MKL-DNN will allocate memory automatically */
    auto conv_src_memory = memory(conv_pd.data.src_primitive_desc);

    /* create a reorder between data, make it an input for a convolution */
    conv_input = reorder(user_src_memory, conv_src_memory)

    /* put the reorder in the net */
    net.push_back(conv_input);
}
```

8. Create a convolution primitive and add it to the net.
```cpp
/* note, that conv_input (whether it is memory or reorder) becomes a
 * dependency for the convolution, so stream won't execute the convolution
 * before this conv_input */
auto conv = convolution(conv_pd, conv_input, conv_weights_memory, conv_user_bias_memory, conv_dst_memory);
net.push_back(conv);
```

9. Finally create a stream, push all the primitives in it and wait for it to
    be done.
```cpp
stream().submit(net).wait();
```

@subpage legal_information

