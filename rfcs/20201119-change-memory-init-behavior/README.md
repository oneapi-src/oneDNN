# Zero padding management in oneDNN

## Problem statement

In oneDNN, we support padded tensors (see padded dims in memory
descriptor).  The implicit contract for primitives is that:
a. they can assume their inputs/outputs are properly zero padded,
b. the zero padding of the destination(s) should be preserved (either
   by explicitly writing zeros or by not modifying the padded area).

In order to guarantee a., we currently zero initialize the padded area
of the memories when the handle is set (either during memory creation,
or when `set_data_handle` is called). This approach has a few caveats:
- since `set_data_handle` also initializes the memory it is currently
  a blocking operation that cannot be done asynchronously. In
  particular, this function  takes no stream as argument and does not
  return an event. So on GPU, we have no other option than do the
  memory initialization in a service stream (part of the engine) and
  wait for it to complete.
- we can have a large overhead of zero padding since framework
  integration cannot keep all memory objects alive, so they recreate
  them and set data handles frequently (see [Appendix A](#appendix-a)).

These issues are magnified as we are using larger blocking sizes to
support new systolic HW (e.g. AMX), so the likelihood of needing
padding increases.

## Goals

1. Reduce or remove the spurious calls to zero padding.
2. Prevent the memory creation and handle setting from being blocking
   operations.
3. Not break external code that compute directly on blocked buffers.

## Options

### Option 1. Move destination zero-padding to primitive execution (preferred)

Here we would change the assumptions that each primitive
implementation can make:
a. The inputs of a primitive are always properly zero padded
b. Each primitive should guarantee its destination(s) are properly
zero padded.

If we consider that all computation on blocked buffer happen in
oneDNN, then initial zero padding would be done by the reorders from
plain to blocked layout, and carried forward by all primitives.

The assumption a. is already guaranteed currently, but b. is stricter:
before, we just had to preserve the destination zero padding in
primitive execution, here we would have to explicitly zero pad. This
is usually not a problem since most our computation naturally
propagate zeros from inputs to the outputs when processed, but some
kernels that support tail handling might not propagate zeros properly
(and this could be easily fixed in general, e.g. by calling GEMM with
padded dimensions, and not with actual dimensions).

There are two main impacts to this option.
- An impact on users that have custom computations on blocked
  buffers. This option is a breaking change if these implementations
  do not propagate zero padding to their destination properly.For
  their code to work, they will have to modify it to properly
  initialize the destination padding to zero. This modification will
  either be in their code logic, or by calling a reorder after their
  computation, at the expense of an extra memory copy.
- An impact on library developers since we have to check that all
  implementation comply with this new set of assumptions. For
  implementations that do not support this assumption, we will have to
  call zero padding on destination.

> **NOTE** Make an experiment to list primitives that already support
> the updated assumptions and those which don't. In particular,
> primitives that call gemm functions might not propagate zero padding
> properly since gemm functions have a proper tail handling.

This option:
- fully achieves goal 1, since all zero pad calls are eliminated
- fully achieves goal 2, since no more initialization happens when
  setting a memory handle
- does not achieve goal 3, but provides a (not so good) workaround.
  Note that if the users custom kernels properly propagate zeros from
  their inputs, then no harm is done and we are all good.

> **NOTE** we could later add a flag to allow primitives to assume
> that their destination is already properly zero padded. This
> situation happens if memory objects are kept alive and reused for
> example. Such a flag would help if we have situation where doing
> tail handling is faster than writing zeros to padding.

### Option 2. Move destination zero-padding to primitive execution, and expose a memory initialization primitive

Here it would be essentially the same as option 1. However, to
simplify the transition for users that have custom computations on
blocked buffers, we would expose a new primitive to initialize memory.
There are two sub-options here:
- option 2.a: introduce an "in-place" reorder. If the user calls
  reorder with the same input and output, the only effect that should
  be expected is explicit zero padding. The bad side of this option is
  that it is somehow a "trick" since this reorder would break
  assumption a. But if this is aiming only advanced users writing
  kernels that compute on blocked buffers, then maybe we can live with
  this documented exception.
- option 2.b: introduce a new `memory_cleanup` primitive, that should
  be used after every custom computation done on a oneDNN memory
  handle. This one has the merit of being more explicit than calling a
  reorder.

Both approaches try to simplify explicit zero padding from users that
write custom kernels. However, it is unclear if they would indeed be
useful, since they require that the component responsible for the
custom kernel call also has access to the oneDNN memory object. This is
the case for iDeep, but it is not clear if other potential users are
in the same situation.

This option:
- fully achieves goal 1, since all zero pad calls are eliminated
- fully achieves goal 2, since no more initialization happens when
  setting a memory handle
- does not achieve goal 3, but provides a provides a simple mitigation
  that should not impact performance too much.  Note that if the users
  custom kernels properly propagate zeros from their inputs, then no
  harm is done and we are all good.

## Option 3. Move source zero-padding to primitive execution

Here we would change the assumptions that each primitive
implementation can make:
a. The inputs of a primitive are not always properly zero padded
b. The outputs of a primitive do not have to be properly zero-padded.

This can reduce the number of calls to zero padding in two ways:
- primitives can skip zero padding on their source if they support
  tail handling
- when the users pass handles between two primitive, it will result in
  one zero pad call instead of two.

This option:
- partially achieves goal 1, since some zero padding calls will be
  avoided (for example, the double zero padding described in [Appendix
  A](#appendix-a)), but not fully since we will have to explicitly
  call zero padding in our implementations (since most of them do not
  currently support tail handling). How often this call will be needed
  depends on how many implementation can return a dirty output.
- fully achieves goal 2, since no more initialization happens when
  setting a memory handle.
- partially achieves goal 3: since source memories are always
  initialized in primitive execution, custom kernels can freely spoil
  the padding of their outputs. However, oneDNN primitives are allowed
  to spoil their outputs, so custom kernels can no more assume their
  inputs properly padded (should be fine for eltwise like functions).

## Option 4. Move zero-padding to primitive execution, and add meta-data to avoid spurious zero padding.
This option uses no assumption on the zero-padding of its input/output
tensors. Here, we would carry this information through meta-data in the
memory descriptors. That meta-data can then be used by primitive
implementations to skip some spurious zero-padding calls.  The
meta-data would be stored in the memory descriptor and would give the
initialization status of the memory with 3 possible states, `clean`,
`dirty` or `unknown`.

The assumptions that each primitive implementation can make are now
implementation dependent, as they can accept whatever they support.
In case the user provides a memory which states does not match what
the primitive expects, a reorder needs to be issued and a reorder will
proceed with the proper memory initialization (namely zero padding).

When a user initializes a memory descriptor (with `memory_desc_init`
or with the memory constructor) the meta-data would default to `clean`
if not applicable, or `unknown` if the tensor is padded (we could
extend the operators that initialize a memory descriptor later to
accept a state from the user).  When the memory descriptor is
initialized by a primitive, it will use whatever state the primitive
implementation expects (`clean` if primitive accepts only clean
memories, `dirty` otherwise).

This option:
- partially achieves goal 1, since some zero padding calls will be
  avoided (for example, the double zero padding described in [Appendix
  A](#appendix-a)), but not fully since we will have to explicitly
  call zero padding in our implementations (since most of them do not
  currently support tail handling). How often this call will be needed
  depends on how many implementation can return a `dirty` output.
  Another caveat is that zero padding could happen in reorder if some
  implementation require a `clean` input but the previous layer
  produces a `dirty` output. This zero pad will be more time and
  memory consuming than today since it will copy the data to another
  buffer.
- fully achieves goal 2, since no more initialization happens when
  setting a memory handle
- partially achieves goal 3, since source memories are always
  initialized in primitive execution, custom kernels can spoil the
  padding of their outputs. However, oneDNN primitives are allowed to
  spoil their outputs, so custom kernels can no more assume their
  inputs properly padded. This could be worked around by calling a
  reorder (as in option 1), or better by modifying all implementations
  to return a `clean` memory (so same work as in option 1).


## Option 5. Keep memory initialization when handle is set but give control to the user

Here we would change the assumptions that each primitive
implementation can make as in option 1:
a. The inputs of a primitive are always properly zero padded
b. Each primitive should guarantee its destination(s) are properly
zero padded.

However, we would keep the zero padding logic when a memory data
handle is set. In order to make setting the handle of a memory
asynchronous, `set_data_handle` would take a stream as input and
return an event (not that for the memory constructor that takes a
handle as input, we would still have to be synchronous since no event
can be returned in this case). The `set_data_handle` operation would
also take a boolean as input, that the user would set to `true` if
memory was already initialized (so if it was the output of a oneDNN
primitive), and false otherwise (default value).

This option:
- partially achieves goal 1. If the user set each memory status
  properly, then the number of calls to zero pad can be minimized (or
  even zero if no custom kernel is called). However, it would require
  a lot of effort on the user side to achieve this. If nothing is done
  on the user side to reduce zero padding, then the situation remains
  unchanged compared to today.
- fully achieves goal 2, since now setting a handle can be done
  asynchronously.
- achieves goal 3. By default, all memories are zero padded, and the
  user has full control to enforce zero padding when calling custom
  kernels.

This option seems tempting since it solves all the issues. But the
very big downsides are that
1. it does not have good performance by default
2. it puts more burden on the user side to extract performance.

Unfortunately, these two points above make the likelihood of that
approach to be effective very small.

## Overall


| option   | who zero-pads                   | which tensor  | implicit zero-pad calls needed | library impl impact | user impact                                                 |
|----------|---------------------------------|---------------|--------------------------------|---------------------|-------------------------------------------------------------|
| current  | memory creation/set_data_handle | all           | n                              | none                | none                                                        |
| Option 1 | primitive execution             | destination   | 0                              | almost none         | none if custom kernels produce clean outputs                |
| Option 2 | primitive execution             | destination   | 0                              | almost none         | none if custom kernels produce clean outputs                |
| Option 3 | primitive execution             | source        | n/2                            | almost none         | none if custom kernels accept dirty inputs                  |
| Option 4 | primitive execution / reorder   | any if needed | [0, n/2]                       | depends             | none                                                        |
| Option 5 | memory creation/set data handle | any if needed | [0, n] (depends on user)       | almost none         | important (need to track padded state for good performance) |


Overall, options 2, 4 and 5 potentially leak an implementation detail
to the API (namely, that memory needs initialization in some
situations).

The recommendation would go to option 1 since it is the only one that
does not leak an implementation detail to the API. If users of custom
kernels ask for it, we can transition to option 2, preferably 2.a
since it would just be a documented exception vs a full primitive.

Option 3 does not enable to completely get rid of zero padding.

Option 4 seems like a good middle ground since it is flexible, but it
has the potential to create future performance issues if some
primitives return dirty outputs, since zero padding would happen
through reorder and not in-place.  However it could be a solid option
if all the implementations return clean outputs (so same as option 1
assumption on outputs), since it would solve all the goals.

I would recommend to stay away from option 5 since it is disruptive to
the framework integration (basically we make them responsible to fix a
performance issue in our library).

## Appendix A. How framework developer pass data between oneDNN primitives
In some frameworks, data between primitive execution is passed by
handle, the oneDNN memory object is not passed directly. Changing that
behavior seem to require non trivial re-factoring on their side.

To illustrate the impact of this design, we assume that we run two
primitives one after the other (p0 and p1). We also assume that no
reorder happens between the two primitive executions (so
p1_pd.src_desc() matches p0_pd.dst_desc()).

Before the execution of each oneDNN primitive, the framework code
creates oneDNN memory objects.  During the creation of these memory
objects, zero padding can happen on all of them, including the
destination tensors.

```c++
...
auto p0_src_mem = memory(p0_pd.src_desc(), x_handle); // (1) zero padding here
auto p0_dst_mem = memory(p0_pd.dst_desc(), y_handle); // (2) zero padding here
...
args.insert({DNNL_ARG_SRC, p0_src_mem});
args.insert({DNNL_ARG_DST, p0_dst_mem});
p0.execute(s, args);
...
```

The same happen when we execute p1.

```c++
```c++
...
auto p1_src_mem = memory(p1_pd.src_desc(), y_handle); // (3) zero padding here
auto p1_dst_mem = memory(p1_pd.dst_desc(), z_handle); // (4) zero padding here
...
args.insert({DNNL_ARG_SRC, p1_src_mem});
args.insert({DNNL_ARG_DST, p1_dst_mem});
p1.execute(s, args);
...
```

The notable things to remark here are:
- zero-padding in (3) is redundant, since the p0 execution already
  guarantees that the memory in y_handle is properly zero padded,
- the zero padding in (1) and (3) is not necessary for primitive
  implementations that support tail handling (e.g. GEMM based
  primitives).
- the zero padding in (2) and (4) is not necessary for primitive
  implementations that carry forward the zero padding from their
  inputs (a lot of operations like convolution and GEMM do that
  naturally).
