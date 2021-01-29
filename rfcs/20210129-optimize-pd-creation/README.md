# Optimization of Primitive Descriptor Creation

## Motivation

There are three majors steps to create a oneDNN primitive:
1. Create operation descriptor
2. Create primitive descriptor
3. Create primitive

In frameworks, all the three steps are performed every time when a user executes
an operation. Typically, the last two steps are the most expensive hence they significantly
affect performance of the operation.

oneDNN recently introduced a cache for primitives to significantly reduce primitive
creation time - the most expensive step. The second most expensive step is creating
the primitive descriptor. The cost may vary from several microseconds to hundreds
of microseconds depending on number of the primitive implementations and time of
primitive descriptor initialization.
Such overhead is significant because it is comparable to the execution time of the corresponding primitive.
This is the motivation to start optimizing the primitive descriptor creation step.

## Proposal

Introducing one more cache for primitive descriptors doesn't look appealing for
the following reasons:
1. Bad user experience because user has to manage two caches
2. Maintenance cost (maintain two multithreaded caches)
3. Increases memory consumption
4. New API to control one more cache

There are two options to reduce time of primitive descriptor creation.

### Option 1: Optimizing `pd_t::init()`

Pros
* Internal optimization
* A stand-alone optimization

Cons:
* Time is not guaranteed and varies for different implementations
* For some implementations (e.g. BRGEMM based primitives) `pd_t::init()` takes
a lot of time to compute block sizes. This still should be done somewhere
* Amount of work is significant. Basically, this approach requires to optimize
each of the `pd_t::init()` functions individually
* Maintenance cost is high. Everybody should be careful with extending `pd_t::init()`
* The `pd_t::init()` is not the only time consuming part:
    * Copy of operation descriptor and attributes (which are extending over time)
    * Copy of memory descriptors. This is required because the copied operation
    descriptor must not be changed hence there is a need in copying memory
    descriptors which contain `any` to replace `any` with a particular format tag

### Option 2: Reusing Primitive Cache

Pros:
* Guaranteed small time if primitive cache is enabled (~2-6 microseconds)
* Internal optimization, can be changed at any time
* Easy to implement
* Maintenance cost is low

Cons:
* Binds caching mechanism with the dispatching mechanism

### Conclusion

Our goal is to reduce operation execution time in frameworks by reducing time
of primitive descriptor and primitive creation. The option 2 can reduce the time
as much as possible. The option cannot compete with guaranteed 2-6 microseconds.

The proposal is to implement an internal mechanism that would significantly
reduce time of primitive descriptor creation but do not expose the mechanism to
user.

### Idea Description

The existing *primitive cache* already contains all the abstractions that are
created during the above 3 steps. The `primitive_t` abstraction contains
`primitive_desc_t` which contains `op_desc_t` and `primitive_attr_t`.
According to the *primitive cache* design there was added an ability to share
the `primitive_desc_t` abstraction with `shared_ptr`. Therefore we can simply
use it to create a new `primitive_desc_t` with minimum overhead.

Since primitive descriptor and primitive are created for each execution of an
operation in frameworks and the approach to caching primitives works it will
also work for primitive descriptors.

## Design

Basically, the cache related work is done as part of enabling the *primitive cache*.
The main goal now is to provide a bridge between primitive descriptors creation API and the *primitive cache*.

### Key for Primitive Descriptor

The key which is used to find primitives in the *primitive cache* can be used to find primitive
descriptors too. However, there are a few problems to solve.

Currently, the key looks as follows:
```cpp
struct key_t {
     key_t(const primitive_desc_t *pd, const engine_t *engine, int impl_nthr);
    // Obtained from `primitive_desc_t`
    op_desc_t *op_desc_;
    primitive_attr_t *attr_;
    std::type_index impl_id_;
    std::vector<memory_desc_t> bwd_mds;

    // Obtained from `op_desc_t`
    primitive_kind_t primitive_kind_;

    // Obtained from `engine_t`
    engine_kind_t engine_kind_;
    runtime_kind_t runtime_kind_;
    device_id_t device_id_;

    // Obtained from ctor
    int impl_nthr_;

    // ... //
```

When creating a key to find a primitive in the *primitive cache* there is a primitive descriptor
which is used to obtain some of the required data. However, in case of primitive descriptor
creation for all primitives except reorder, concat and sum user creates `op_desc_t`,
`primitive_attr_t`, `engine_t` and optionally a hint primitive descriptor. Therefore
`impl_id_`, `bwd_mds` and `op_desc_t` for the aforementioned primitives cannot
be obtained at that point.

#### Implementation ID

The rationale behind having an implementation ID is to be able to differentiate
the same primitives which have different underlying implementations. Such primitives
can be created when user uses primitive descriptor iterator to manually iterate
through the list of implementations.

The problem here is that the primitive descriptor is only going to be created and
currently there is no way to get implementation ID without creating primitive
descriptor.

Currently, `std::type_index` of a primitive descriptor of a particular implementation
is used as the implementation ID. In general, this approach has only one minor issue,
it doesn't make it possible to build oneDNN without RTTI support, though it's
unclear whether such ability is needed. However, to be able to use the existing
key and hence the existing *primitive cache* to speed up primitive descriptor
creation the implementation ID must be reworked.

The idea is to replace the existing implementation ID with a new one which is
defined as follows:

Implementation ID is a relative index of a valid implementation for the given
operation descriptor, attributes and a forward primitive descriptor hint.

For example there are operation descriptor and attributes for which some of
the implementations in the implementation list are valid. Then the implementation
IDs will be the following:

| Implementation ID | Implementation index in the list | Valid |
| ----------------- | -------------------------------- | ----- |
|       -           |                0                 |   -   |
|       0           |                1                 |   +   |
|       -           |                2                 |   -   |
|       1           |                3                 |   +   |
|       -           |                4                 |   -   |
|       2           |                5                 |   +   |

In other words, the implementation IDs are the indexes in a hypothetical list
of valid implementations for the given operation descriptors and attributes.
The implementation ID is represented as primitive descriptor iterator offset.
Every time the iterator is incremented the offset is incremented too. In other words,
the offset is how many time the iterator was incremented.

The implementation ID can be non-zero only when using a primitive descriptor
iterator to iterate through the list of implementations. Each attempt to iterate
through the list to get the next valid implementation increments the implementation
ID. In all other cases the implementation ID will be 0.

Also, there are 3 primitives that do not support the iterator, those are sum,
concat and reorder, therefore the implementation ID for those primitives is always
0.

The implementation ID will be stored in `primitive_desc_t` so that it can be
obtained to create a key during primitive creation.

#### Automatic Algorithm Selection for Convolution

oneDNN supports `dnnl::algorithm::convolution_auto` algorithm that instructs
the library to automatically select the best algorithm based on the heuristics
that take into account tensor shapes and the number of logical processors
available. The algorithm is specified for convolution operation descriptor.

oneDNN provides a query mechanism for getting information from oneDNN abstractions.
For example a user can query a primitive descriptor from created primitive. oneDNN
guarantees that the primitive descriptor that the user used to create the primitive
is equal to the the primitive descriptor that the query mechanism returns. This is
guaranteed not only for primitive descriptors but for all other abstractions.
However, there is an exception - operation descriptor for convolution. The thing
here is that if the user created primitive descriptor for convolution using
an operation descriptor with algorithm `convolution auto` the query mechanism
will return the operation descriptor with the algorithm that the library selected.
This happens because oneDNN modifies the operation descriptor when creating
primitive descriptor.
Given that, `op_desc_t` that a user uses to create primitive descriptor and
`op_desc_t` which is obtained from primitive descriptor to create a key to find
the requested primitive in the *primitive cache* may differ in case of convolution therefore
they cannot be compared as is.

The idea to rely on the implementation ID to distinguish the algorithm and ignore
the field in operation descriptor during comparison.

#### Forward Primitive Descriptor Hint

For some primitives the hint can affect equality of primitive descriptors.
Currently, there are only two such primitives: pooling and shuffle. Depending on the
hint the primitive descriptors can have different `diff_src` and `diff_dst`.
To differentiate such primitives in the primitive cache the key contains
corresponding memory descriptors. For pooling and shuffle the key contains a
vector which contains `diff_src` and `diff_dst` which are obtained
from the primitive descriptor.

When creating a key to find primitive descriptor in the primitives cache the
hint can be used to deduce `diff_src` and `diff_dst` memory descriptors, or any
other memory descriptors that affect equality.

The idea is to introduce the following function:

```cpp
struct primitive_desc_t {
    virtual std::vector<memory_desc_t> hint_mds(bool is_hint) const;
};
```
This function returns a vector of memory descriptors that might affect the
equality of primitive descriptors for backward pass.

This function is used for creating a key to fetch primitive or primitive descriptor
from cache.
Depending on whether it is called from a hint or not there are two scenarios:
1. When creating a primitive descriptor for backward pass there may be a forward
primitive descriptor hint that can be used to obtain the memory descriptors. In
this case the `is_hint` argument must be `true`.
2. When creating a primitive this function is called from a non-hint primitive
descriptor that can be either forward or backward. In this case the `is_hint`
argument must be `false`.
    * For forward it will return an empty vector.
    * For backward it will return a vector of memory descriptors if the implementation
    depends on a forward primitive descriptor.

The current primitives that override `hint_mds` are:
* pooling
* shuffle

Later the list of primitives can be extended. For instance, currently there is no
convolution on the list because nthrs + op_desc (even with format=`any`) + attributes
fully define a particular implementation.

#### Operation Descriptor for Sum, Concat and Reorder

The difference between these primitives and others is that they don't have operation
descriptors. Since operation descriptor is part of the key all primitives have to
have operation descriptors. To fulfil the requirement there were introduced
internal operation descriptors to implement primitive cache.

```cpp
struct dnnl_reorder_desc_t {
    dnnl_primitive_kind_t primitive_kind;
    dnnl_memory_desc_t src_md;
    dnnl_memory_desc_t dst_md;
    dnnl_engine_kind_t src_engine_kind;
    dnnl_engine_kind_t dst_engine_kind;
};

struct dnnl_concat_desc_t {
    dnnl_primitive_kind_t primitive_kind;
    dnnl_memory_desc_t dst_md;
    dnnl_dim_t n;
    dnnl_dim_t concat_dimension;
    std::vector<dnnl_memory_desc_t> src_mds;
};

struct dnnl_sum_desc_t {
    dnnl_primitive_kind_t primitive_kind;
    dnnl_memory_desc_t dst_md;
    dnnl_dim_t n;
    std::vector<float> scales;
    std::vector<dnnl_memory_desc_t> src_mds;
};
```

These operation descriptors are created and filled in constructors of primitive
descriptors for the corresponding primitives.

The main problem with these operation descriptors is that they have to be created
to create a key to find primitive descriptor in the cache of primitives. It brings
unnecessary overhead.
Since the primitive descriptors take ownership of all the data that is used to create
the internal operation descriptors there is no need to copy the data. The idea
is to replace deep copy with shallow copy for all data fields which are POD or
class types. This would allow to create the operation descriptors for the key
without undesired overhead.

The updated operation descriptors can look as follows:

```cpp
struct dnnl_reorder_desc_t {
    dnnl_primitive_kind_t primitive_kind;
    const dnnl_memory_desc_t *src_md;
    const dnnl_memory_desc_t *dst_md;
    dnnl_engine_kind_t src_engine_kind;
    dnnl_engine_kind_t dst_engine_kind;
};

struct dnnl_concat_desc_t {
    dnnl_primitive_kind_t primitive_kind;
    const dnnl_memory_desc_t *dst_md;
    dnnl_dim_t n;
    dnnl_dim_t concat_dimension;
    const dnnl_memory_desc_t *src_mds;
};

struct dnnl_sum_desc_t {
    dnnl_primitive_kind_t primitive_kind;
    const dnnl_memory_desc_t *dst_md;
    dnnl_dim_t n;
    const float *scales;
    const dnnl_memory_desc_t *src_mds;
};
```

### Multithreading

Since the cache for primitives is multithreaded there is no need to implement
anything additionally.
