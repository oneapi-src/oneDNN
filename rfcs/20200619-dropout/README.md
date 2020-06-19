# Proposal for oneDNN dropout primitive

## Introduction

The main motivation is request from Apache MXNet community:
https://github.com/oneapi-src/oneDNN/issues/656. Currently MXNet is optimized
with viRngBernoulli from VSL. The problem is that viRngBernoulli returns int*
where one element has typically 4 bytes. They created new requirement in order
to reduce memory space of mask. Mask has boolean values {0, 1}, so instead of
integer representation(4 bytes) it can be stored as 1 bit in bit mask (similar
optimization is used for specialization of std::vector<bool> in C++ standard
library). In order to achieve the requirement of storing mask as bit mask, they
need to perform conversion from table of int to bit mask which reduces memory,
but has additional performance impact (https://github.com/apache/incubator-mxnet/pull/16735),
that's why they asked for dropout primitive.

### Current state

Currently in frameworks that use ONE DNN, dropout is implemented as chain of
elementwise operations.

### Implementation in frameworks

All mainstream frameworks dropout implementations are based on paper:
http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf.

output = input * mask * scale,

where mask is created based on Bernoulli distribution, with probability 1 - rate,

where rate is fraction of input units to drop (real value between 0 and 1),
usually default value is 0.5,

where scale is 1 / (1 - rate)

#### MXNET [docs](https://beta.mxnet.io/api/gluon/_autogen/mxnet.gluon.nn.Dropout.html)

```python
mxnet.gluon.nn.Dropout(rate, axes=(), **kwargs)
```
**Parameters**

- rate : float - Fraction of the input units to drop. Must be a number between
0 and 1.
- axes : tuple of int, default () - The axes on which dropout mask is shared. If
empty, regular dropout is applied.

Axes usage analysis example:

- Data format: NCHW
- Axes: (0, 2)
- Result mask shape (1, C, 1, W)

Next mask is brodcasted to input shape. Logical result size mask is equal to
NCHW N*H*result_mask shape. In that case the benefit is that we used N*H less
memory for mask shape.

#### Tensorflow [docs](https://www.tensorflow.org/api_docs/python/tf/nn/dropout)

```python
tf.nn.dropout(x, rate, noise_shape=None, seed=None, name=None)
```
**Parameters**

- xrate : tensor - A floating point tensor.
- rate: scalar tensor - A scalar Tensor with the same type as x. The probability
that each element is dropped. For example, setting rate=0.1 would drop 10% of
input elements.
- noise_shape: tensor - A 1-D Tensor of type int32, representing the shape for
randomly generated keep/drop flags.
- seed: integer - A Python integer. Used to create random seeds. See
tf.random.set_seed for behavior.

Noise shape analysis example:
Data format: NCHW
- Noise shape: [1, C, 1, W]
- Result mask shape (1, C, 1, W)

- Noise shape: [1, C, 1, W/2]
- Result mask shape: exception not brodcastable, noise shape invalid.

The way of working is exactly the same as in MXNET. One could think that if input
is tensor [4,10], noise shape [2,10] will be broadcastable, but only applicable
values are exact dimensions or ones. In case input tensor [4,10] it will be
[1,10] or [4,10] or [4,10].

#### Caffe [docs](https://caffe.berkeleyvision.org/tutorial/layers/dropout.html)

```cpp
message DropoutParameter {
  optional float dropout_ratio = 1 [default = 0.5];
}
```
**Parameters**
- dropout_ratio : float - Fraction of the input units to drop. Must be a number
between 0 and 1.

#### PyTorch [docs](https://pytorch.org/docs/master/generated/torch.nn.Dropout.html)

```python
torch.nn.Dropout2d(p=0.5, inplace=False)
torch.nn.Dropout3d(p=0.5, inplace=False)
```
**Parameters**
- p : (float, optional): probability of an element to be zero-ed.
- inplace: (bool, optional): If set to True, will do this operation in-place

#### Summary
The only common part for all frameworks is dropout ratio. Extra parameters
introduced by Mxnet and Tensorflow accordingly named axes and shape introduce
extra mask memory usage optimization. What is more tensorflow user can specify
seed for random generator and pytorch user can enforce inplace computation.

## Proposal

### Option 1 Standalone dropout primitive

```cpp
typedef enum {
  dnnl_mcg31m1,
  dnnl_mt19937
} dnnl_rng_algorithm_t

// A descriptor of a dropout operation.
typedef struct {
      /// The kind of primitive. Used for self-identifying the primitive
      /// descriptor. Must be #dnnl_dropout.
      dnnl_primitive_kind_t primitive_kind;
      // The kind of propagation. Possible values: #dnnl_forward_training, #dnnl_backward
      dnnl_prop_kind_t prop_kind;
      // Source and destination memory descriptor.
      dnnl_memory_desc_t data_desc;
      // Source and destination gradient memory descriptor.
      dnnl_memory_desc_t diff_data_desc;
      // Seed memory descriptor.
      dnnl_memory_desc_t seed_desc;
      // Algorithm for random number generation
      dnnl_rng_algorithm_t rng_algorithm;
      // Noise shape for randomly generated keep/drop flags
      dnnl_dims_t noise_shape;
      // Dropout ratio fraction of input units to drop (value between 0 and 1)
      float ratio;
} dnnl_dropout_desc_t;

// Extend ARG define list
#define DNNL_ARG_SEED number
```
Properties:
- user is responsible for generating seed. Usually seed will be tensor with
size=1. User passes seed in execute using DNNL_ARG_SEED. Primitivie in
DNNL_ARG_SEED memory returns end-state of rng engine. In next calls of 'execute'
user is deciding whether to generate new seed or use seed from end state of rng;
- generated mask stored in workspace as bit mask - once generated in forward pass,
reused in backward pass;

Pros:
- mask stored internally in workspace, we can choose internally any format for it,
for example bit mask requested by mxnet community, no need to add bit_mask
data type in API;
- support for additional specific dropout params (noise_shape from tensorflow,
axis mxnet);
- use of internally stored mask, creates possibility of further optimizations,
minimizing size of the mask in case noise shape applied (logical mask
broadcasting);
- this approach limits number of read-from memory - one read (src data), two
writes (generated mask, dest data);

Cons:
- poor flexibility (hard to reuse RNG implementation in other possible use cases);
- api extension (another option in api);

### Option 2 Standalone random number generator

```cpp
typedef enum {
  dnnl_mcg31m1,
  dnnl_mt19937
} dnnl_rng_algorithm_t

// A descriptor of a rng operation.
typedef struct {
      /// The kind of primitive. Used for self-identifying the primitive
      /// descriptor. Must be #dnnl_rng.
      dnnl_primitive_kind_t primitive_kind;
      // Destination memory descriptor.
      dnnl_memory_desc_t data_desc;
      // Seed memory descriptor.
      dnnl_memory_desc_t seed_desc;
      // Algorithm for random number generation
      dnnl_rng_algorithm_t rng_algorithm;
} dnnl_rng_desc_t;

// Extend ARG define list
#define DNNL_ARG_SEED number
```
Properties:
- dropout realized in following way: mask generated by RNG primitive stored to
memory is one of the inputs of binary mul while the second input is dropout src
data (data to be masked). Binary has eltwise operation - postops which provides
scaling demanded by dropout;
- user is responsible for generating seed. Usually seed will be tensor with
size=1. User passes seed in execute using DNNL_ARG_SEED. Primitivie in
DNNL_ARG_SEED memory returns end-state of rng engine. In next calls of 'execute'
user is deciding whether to generate new seed or use seed from end state of rng;

One of the RNG parameters should be distribution type (for example Bernoulli in
case of dropout) which can in general accept different number of arguments. For
Bernoulli is only one ratio, for others there are multiple of them. With that
said we have to be ready for taking various sets of arguments. We should consider
introducing extra postop - distribution for RNG primitive.

Pros:
- flexibility - primitive can be used everywhere when random input is needed
Cons:
- rng must be able to generate bitmask, currently in oneDNN we don't have any
possibility to described memory passed by user as bitmask. Introduction of
support for new data type will affect potentially wider scope primitives, in
case of dropout binary primitive should be able to work using bit mask type;
- implementation would demand one write to memory frm rng, a then to reads from
in binary (mask, dropout_src) and final write memory to dst (2x read + 2 write)

### Summary

After analysis of two proposed options, we recommend developing "Option 1
Standalone dropout primitive". It provides better performance - 1 read less -
covering Mxnet community demand. It is more straightforward in matter of
development (we don't introduce new memory type - bitmask, we are supporting
only one Bernoulli distribution) and usage one api call vs two api calls.
The main pros of standalone RNG is flexibility of this solution, but it's
common usability is questionable.

### Random number generation

The main challenge of the implementation is to create random number generation
in one dnn. Such implementation should have the following properties:
- efficiency - implementation in one dnn should be at least as efficient as
viRngBernoulli from VSL and optimized for usage of multithreading and sse4.1,
avx2, avx512
- testability - by setting the same seed we should be able to create exactly the
same mask using benchdnn and one dnn optimized implementation

In case of calculation reference output in tests by benchdnn, mask generation
with bernoulli distribution can be achieved by using C++ standard library. Since
C++11 bernoulli_distribution is part of C++ standard library, by utilizing
discard function and omp we can achieve mulithreaded effect as in example below:

**C++ reference single thread**
```cpp
#include <vector>
#include <random>
#include <algorithm>
#include <experimental/iterator>
#include <iostream>
#include <cmath>

int main()
{
    const int seed = 10;
    const float p = 0.5;
    const std::size_t mask_size = 25;
    std::vector<bool> mask(mask_size);

    std::mt19937 engine(seed);
    std::bernoulli_distribution gen(p);
    auto generator = [&](...) -> bool { return gen(engine); };
    std::transform(mask.begin(), mask.end(), mask.begin(), generator);

    std::copy(mask.begin(), mask.end(), std::experimental::make_ostream_joiner(std::cout, ", "));
}
```

**C++ reference multithreaded**

```cpp
#include <vector>
#include <random>
#include <algorithm>
#include <experimental/iterator>
#include <iostream>
#include <cmath>
#include <omp.h>

template <typename T>
void generate(size_t min, size_t max, T & collection, int seed, float p)
{
    std::mt19937 engine(seed);
    std::bernoulli_distribution gen(p);

    auto b = std::numeric_limits<long double>::digits;
    auto R = engine.max() - engine.min() + 1;
    auto discard_value = std::max(1.0, std::ceil(b/std::log2(R)));
    engine.discard(min*discard_value);
    const auto generator = [&](...) -> bool { return gen(engine); };
    std::transform(collection.begin()+min, collection.begin()+max, collection.begin()+min, generator);
}

int main()
{
    const int seed = 10;
    const float p = 0.5;
    const std::size_t mask_size = 25;
    std::vector<bool> mask(mask_size);

    #pragma omp parallel num_threads(5)
    {
        int tid = omp_get_thread_num();

        generate((tid*5), (tid*5) + 5, mask, seed, p);
    }

    std::copy(mask.begin(), mask.end(), std::experimental::make_ostream_joiner(std::cout, ", "));
}
```

Both single threaded and multithreaded versions will output exactly the same
values (compiled using gcc8.3). For example Bernoulli distribution implementation
for gcc is defined
https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/random.h.
If we go the way of using pure c plus plus for calculating reference by benchdnn
in tests, we should consider implementation our own internal version of bernoulli
distribution. Implementation Bernoulli distribution in gcc is sampling random
values from uniform distribution and thresholding with a parameter. Concept of
distribution consists of implementation operator(), min(), max() methods, so it
is possible to implement version compatible with standard. There are different
implementations of the same rng engines algorithms in different glibc
versions/implementations, so to get the same results every time - internal
implementation both engine and distribution will be required.

### Random number generation - vectorized implementation

VSL implements several basic RNG engines including VSL_BRNG_MCG31, which is used
by default in mxnet's dropout (see:
https://github.com/apache/incubator-mxnet/blob/master/src/operator/nn/dropout-inl.h line 108).
The engine mentioned performs computation of the following recursive formula
(see: https://en.wikipedia.org/wiki/Linear_congruential_generator):

```python
def lcg(modulus, a, c, seed):
    """Linear congruential generator."""
    while True:
        seed = (a * seed + c) % modulus
        yield seed
```

In VSL MCG31 engine is implemented slightly different - it skips "c" parameter
(consulted with MKL team). With this change, the core of the formula simplifies
to:

```python
seed = (a * seed) % modulus
```

It is worth noting that in such case:

```python
seed[1] = seed[0] * A % modulus
seed[2] = seed[0] * (A * A % modulus) % modulus = seed[1] * A % modulus
```

So, first N random values can be computed as a multiplication of vector of
initial seeds (scalar seed broadcasted to vector) by vector of consecutive powers
of "a" modulo "modulus".
The latter vectors of random values, as they depend on the former ones, can be
computed by vector multiplication of previously generated output seed and a
broadcasted power of a modulo "modulus".

With that said, we can introduce a following vectorized formula for computing
vectors of random values (with utilization of SIMD instructions):

```
N - lenght of AVX vector

// first iteration (i=0)
seed_t[0:N] = [seed, seed, ..., seed]
a_t[0:N] = [1, A % modulus, A^2 % modulus, ..., A^(N-1) % modulus]
out[0:N] = seed[0:N] * a_t[0:N] % modulus

//following iterations
aa_t[0:N] = [A^(N-1) % modulus, A^(N-1) % modulus, ..., A^(N-1) % modulus]
for(i=1:...)
    out[N*i:N*(i+1)] = out[N*(i-1):N*i] * aa_t[0:N] % modulus
```

According to GPU team opinion (analysis made by Chereshnev, Eugene) the above
algorithm should be fairly easily portable to GPU.

### RNG State

Should we store RNG state somewhere ? oneDNN primitives should be stateless.

Possible use case:
```cpp
primitve rng(...);
rng.execute(...) ;
rng.execute(...) // second exeucte should start from end state of previous execute;
```
Let's consider https://github.com/apache/incubator-mxnet/blob/master/src/operator/nn/dropout-inl.h
since line 90. In each time they are creating new stream vsl stream and use
viRngBernoulli. They don't save stream and viRngBernoulli doesn't save any state.
The only thing that they're doing is gnenerating new seed in each param.

Possible solution:
- User should pass seed in execute instead primitive_descriptor (in intial draft,
seed was passed in the primitive descriptor)
- User passes seed in execute using DNNL_ARG_SEED. Primitivie in
DNNL_ARG_SEED memory returns end-state of rng engine. In next calls of 'execute'
user is deciding if he wants to use any seed of his choice or use returned seed
from last run and start execution from end-state of previous.

## Open questions

1. Should user be allowed to specify noise shape / axes  ?

In our opinion yes. It is used in two popular frameworks - tensorflow, mxnet.
Rationale for that is further decrease of memory usage + if we ommit now that
parameter and if in the future there will be request to support that, it will
require either add some kind v2 version in api or abi break.

2. Should we support inplace computation (pytorch) ?

In our opionion we can support that, it is not big implementation effort.

3. Do we understand how differences in RNGs affect training results ?

4. In what other primitives/layers/activation_function other than dropout RNG is
used ?

Tensorflow pov:
* one prominent example: random uniform distribution is used for initialization;
* grep for “random” word inside [kernels implementing tf ops](https://github.com/tensorflow/tensorflow/tree/9590c4c32dd4346ea5c35673336f5912c6072bf2/tensorflow/core/kernels) shows following
list:
  - batch_kernels.cc
  - candidate_sampler_ops.cc
  - clustering_ops.cc
  - concat_lib_cpu.cc
  - cudnn_rnn_ops.cc
  - cwise_op_gpu_random_grad.cu.cc
  - cwise_op_random_grad.cc
  - decode_padded_raw_op.cc
  - fixed_length_record_reader_op.cc
  - fractional_avg_pool_op.cc
  - fractional_max_pool_op.cc
  - fractional_pool_common.cc
  - map_stage_op.cc
  - multinomial_op.cc
  - multinomial_op_gpu.cu.cc
  - parameterized_truncated_normal_op.cc
  - parameterized_truncated_normal_op_gpu.cu.cc
  - partitioned_function_ops.cc
  - random_binomial_op.cc
  - random_crop_op.cc
  - random_op.cc
  - random_op_gpu.cu.cc
  - random_poisson_op.cc
  - random_shuffle_op.cc
  - random_shuffle_queue_op.cc
  - range_sampler.cc
  - record_input_op.cc
  - record_yielder.cc
  - sample_distorted_bounding_box_op.cc
  - sdca_internal.cc
  - stateful_random_ops.cc
  - stateful_random_ops_gpu.cu.cc
  - stateless_random_ops.cc
  - whole_file_read_ops.cc
  - word2vec_kernels.cc
* ops seems to be using RNG seriously: word2vec, fractional_max_pool_op.cc,
fractional_avg_pool_op.cc, fractional_pool_common.cc
* following statements are now under verification:
  - Initialization is not on critical path (are there cases when it is?)
  - Fractional pooling also uses RNGs only during op initialization
  - Word2vec is run only during preprocessing and is not on critical path either


EOD

