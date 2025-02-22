# Cold Cache Feature

Benchdnn provides a mode to measure the performance of a desired
functionality. It runs the same execute call with the same memory objects over
and over until specific criterion is no longer triggered, signaling it's time
to stop.

Running measurements this way is advantageous in that it shows almost pure
kernel performance, particularly under conditions where all memory objects may
end up in a system cache. This means much faster loading time, depending on
the cache data level, and better FLOPS or milliseconds for a
given problem. However, it doesn't necessarily reflect the performance that
end users typically have. For example, given an application's number of blocks
and their sizes, most memory objects come from RAM and only some
pieces roll over from the previous layer.

In order to measure the library functionality closer to real-world
applications, benchdnn has introduced a cold cache feature. In the core,
each execution call in a performance measurement loop uses new memory objects
for specific execution arguments, which is controlled by the user.

To enable cold cache, users must specify `--cold-cache=MODE` and choose a
`MODE` from one of three options: `wei`, `all`, or `custom` (lower case).

`wei` mode has benchdnn prepare a pile of weights tensors and, for each new
run, use a new set until the stack is over. Then, it starts from the top of
the pile again. It ensures that pile size is sufficient enough to evict
previously used sets of weights from caches. This mode emulates a regular
situation in topologies for primitives such as (de-)convolutions, inner
product, and matmul. Usually, these primitives (and whole models) use a
technique called "double buffering," where all activations transfer between
two buffers: one for source and one for destination. It allows for an
increased cache hit rate; only weights are taken from RAM. Note that if a
primitive requests this mode but does not have a notion of weights, a warning
is printed to stdout and cold cache is not enabled.

`all` mode estimates sizes for the whole problem and makes equal piles of
memory objects for each execution argument so that the problem executes a
unique set every time. It targets situations when first load happens, such as
reorders. It also targets branching when just two buffers is not enough, and a
copy of the previous layer comes from RAM. Note that the `user` mode scratchpad
counts towards the estimated size since this is a separate memory object. In
`library` mode, it doesn't count because the memory buffer is not under user's
control.

`custom` mode is a compromise between flexibility and command-line usability.
In terms of functionality, it provides the user with an ability to specify the
desired execution arguments to be put in cold cache, but they must be
programmed in benchdnn. Once the `custom` value is provided and the problem
gets executed, benchdnn checks if custom arguments are filled (by default they
are empty). If not, an error is returned to the user with the file name and
line where modifications are expected. Once updated, `custom` mode starts
working with the specified arguments.

Since cold cache targets measurements to show real RAM bandwidth, our
recommendation is to utilize a custom performance template that contains
bandwidth metrics: `--perf-template=%-Gbw%,%0Gbw%`. This example provides both
best and average bandwidth to inspect since the execution cannot control or
guarantee that each run will somehow utilize the cache.
