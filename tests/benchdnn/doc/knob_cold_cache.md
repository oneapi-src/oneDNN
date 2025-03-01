# Cold Cache Feature

## Introduction
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

To measure the library functionality closer to real-world applications, benchdnn
has introduced a cold cache feature. In the core, each execution call in a
performance measurement loop uses new memory objects for specific execution
arguments, which is controlled by the user.

## Usage
```
    --cold-cache=MODE[+EXTENSION]
```

`MODE` can be the one of these options: `none` (the default), `wei`, `all`, or
`custom` (lowercase).

`wei` mode has benchdnn prepare a pile of weights tensors and, for each new
run, use a new set until the stack is over. Then, it starts from the top of
the pile again. It ensures that pile size is sufficient to evict previously used
sets of weights from caches. This mode emulates a regular situation in
topologies for primitives such as (de-)convolutions, inner product, and matmul.

It is common in modern frameworks that a primitive output activations reuse
the same memory pages and cache lines as the input activations of the previously
run primitive. As a result, cache hit rate is often increased for activation
accesses while weights are taken from RAM. Note that if a primitive requests
this mode but does not have a notion of weights, a warning is printed to stdout
and cold cache is not enabled.

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

## Extensions
In addition to the main cold cache mode, extensions may be added. Extensions are
separated with `+` delimiter from the `MODE` and from each other.

### Cold TLB
One of the extensions simulates a cold state for translation lookaside buffer,
or TLB. TLB acts as another level of cache for page table entries. Having
specific memory pages cached in TLB can result in better problem performance.
Once enabled, benchdnn will allocate a specific amount of memory to make cold
cache buffers trigger TLB misses.

The following is the extension syntax:
```
    tlb[:SIZE]
```

`SIZE` is a string-literal consisting of a floating-point number followed by
an `M` (for Megabytes) or `G` (for Gigabytes) character. By default, the `SIZE`
is 1 Gigabyte.

For example, `--cold-cache=all+tlb:2G` will instruct benchdnn to enable cold
cache for all execution arguments and allocates an additional 2 GB of memory
before having a performance run over cold cache arguments.

