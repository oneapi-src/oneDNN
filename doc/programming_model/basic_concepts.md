Basic Concepts {#dev_guide_basic_concepts}
==========================================

### Primitives

Intel(R) MKL-DNN is built around the notion of a *primitive* (@ref
mkldnn::primitive), a functor object that encapsulates a particular
computation such as forward convolution, backward LSTM computations, or a data
movement operation that changes the way data is laid out in memory. A single
primitive can sometimes represent more complex, *fused*, computations such as
a forward convolution followed by a ReLU.

The most basic difference between a *primitive* and a *function* is that a
*primitive* can store **state**. For example, a convolution primitive stores
operation parameters like tensor shapes and can pre-compute other secondary
parameters like cache blocking. This also allows pre-generating code
specifically tailored for the operation to be performed. The time it takes to
perform the pre-computations can be amortized by reusing the same primitive
to perform computations multiple times.

In addition to encapsulating the state, a primitive can also use a *scratchpad*,
a temporary memory buffer, that it uses during computations. The scratchpad
can be either tied to a particular primitive object (which makes that object
non-thread safe), or provided as one of the parameters during execution.

### Engines

*Engine* (@ref mkldnn::engine) is an abstraction of a particular computational
device. Currently, the only supported engine is CPU. Most primitives are
created to perform computations on a particular engine. The only exception is
reorder primitive, which can be used to transfer data between different engines.

### Streams

*Streams* (@ref mkldnn::stream) encapsulate execution context tied to a
particular engine.

### Memory Objects

*Memory objects* (@ref mkldnn::memory) encapsulate engine-specific memory
handles, tensor dimensions, data type, and memory format -- the way tensor
data is laid out. (Side note: Formally, primitives should also be referred to
as *primitive objects*, but because the word 'primitive' is less overloaded than
'memory', we can omit the 'object' part without causing confusion.)

Memory objects are passed to primitives during execution via a special map
that defines which tensor (source, destination or weights, or their gradients)
each memory object represents.

## Levels of Abstraction

Intel MKL-DNN has multiple levels abstractions for primitives and memory objects
in order to expose maximum flexibility to its users.

On the *logical* level, the library provides the following abstractions:

* *Memory descriptor* (@ref mkldnn::memory::desc) provides a way to describe a
  tensor's logical dimensions, data type, and, potentially, the format in which
  the data is laid out in memory. A memory descriptor can be created with a
  placeholder format tag (@ref mkldnn::memory::format_tag::any), which is used
  to indicate that the actual format will be defined later (see
  @ref cpu_memory_format_propagation_cpp).

* *Operation descriptors* (one for each for each supported primitive) serve a
  purpose similar to the memory descriptor: they provide a way to describe an
  operation's most basic properties without specifying, for example, which
  engine will be used to compute them.

* *Primitive descriptors* (@ref mkldnn_primitive_desc_t; in the C++ API there
  are multiple types for each supported primitive) are at an abstraction level
  in between operation descriptors and primitives and can be used to inspect
  details of a particular primitive implementation like expected memory
  formats via *queries* without having to fully instantiate a primitive. The
  primitive descriptors are crucial to implement memory format propagation.

| Abstraction level        | Memory object     | Primitive objects    |
|--------------------------|-------------------|----------------------|
| Logical description      | Memory descriptor | Operation descriptor |
| Intermediate description | N/A               | Primitive descriptor |
| Implementation           | Memory object     | Primitive            |

## Creating Memory Objects and Primitives

### Memory Objects

Memory objects are created from the memory descriptors. It is not possible to
create a memory object from a memory descriptor that has memory format set to
#mkldnn::memory::format_tag::any.

There are two common ways for initializing memory descriptors:

* By using @ref mkldnn::memory::desc constructors or by extracting a
  descriptor for a part of a tensor via
  @ref mkldnn::memory::desc::submemory_desc

* By *querying* an existing primitive descriptor for a memory descriptor
  corresponding to one of the primitive's parameters (for example, @ref
  mkldnn::convolution_forward::primitive_desc::src_desc).

Memory objects can be created with a user-provided handle (a `void *` on CPU),
or without one, in which case the library will allocate storage space on its
own.

### Primitives

The sequence of actions to create a primitive is:

1. Create an operation descriptor via, for example, @ref
   mkldnn::convolution_forward::desc. The operation descriptor can contain
   memory descriptors with placeholder
   [format_tag::any](@ref mkldnn::memory::format_tag::any)
   memory formats if the primitive supports it.

2. Create a primitive descriptor based on the operation descriptor and an
   engine handle.

3. Create a primitive based on the primitive descriptor obtained in step 2.
