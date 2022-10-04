Proposal to introduce max concurrency knobs for threadpool

# 1. Introduction

Historically, oneDNN primitives on CPU were optimized for OpenMP
threading runtime. An inherent property of the OpenMP threading
paradigm is that users can control the maximum concurrency to use
during the program execution (omp\_num\_threads). As a result:
- most of oneDNN CPU primitives are optimized assuming that the
  maximum concurrency is properly set by the end user.
- oneDNN relies on global functions to query the maximum concurrency
  during primitive creation.

Fast forward a few years, oneDNN now supports TBB and Threadpool
runtimes on CPU. Where TBB has global controls to query the maximum
concurrency of the active arena that oneDNN can use during primitive
creation, Threadpool runtime do not have those controls. As a
consequence, performance can be suboptimal.

# 2. Deeper dive into the issue

Here the key ingredients that make Threadpool performance suboptimal are:
- most oneDNN CPU implementation use the maximum concurrency at
  creation time to optimize load balancing.
- oneDNN does not know the maximum concurrency during primitive
  creation time for threadpool runtime
- most oneDNN CPU implementations spawn maximum concurrency tasks in
  their parallel sections.

Improving oneDNN performance with Threadpool runtime would imply
fixing one or more of the above issue, let's discuss them in more details.

## The necesity of knowing maximum concurrency at creation time

One could think that we could optimize without knowing the maximum
concurrency at creation time. However, this information is important
and is used to select blocking sizes in order to achieve good load
balancing (all threads are busy about the same amount of cycles).

For best performance, oneDNN typically uses those blocking sizes as
parameter to jitted kernels. Once the primitive is created, those
kernels are immutable and are tied to specific blocking sizes.

Removing the knowledge of the maximum concurrency at primitive
creation time would hence impact load balancing at execution and
efficiency of jitted kernels. As such, we will now assume that knowing
the maximum concurrency at creation time is actually a necessity for
oneDNN.

## How to know the maximum concurrency at creation time

As mentioned earlier, when using Threadpool runtime, oneDNN does not
have access to a global function to query the maximum concurrency are
primitive creation time.

To circumvent this limitation, oneDNN relies on a simple heuristic to
cap the maximum concurrency to the number of physical cores on a
socket during primitive creation when using Threadpool runtime. This
has two major drawbacks though:
- when the number of threads in the actual threadpool used during
  execution does not divide the number of cores on a socket, it leads
  to major load imbalance issues.
- when the number of thread in the actual threadpool used during
  execution is greater than the number of cores on a socket (e.g. on
  multi socket runs), this can lead to underutilizing the available
  HW and suboptimal performance.

If Threadpool does not provide a way to query the maximum concurrency
(and it can't as there is no notion of task\_arena like TBB), oneDNN
could provide APIs for the users to provide a max\_concurrency at
primitive creation time.

## Remove the tight dependency between maximum concurrency and number of tasks spawned at execution.

This would likely be the most impactful change oneDNN could do.
Because of its inheritage from OMP, most oneDNN implementations always
spawn as many tasks as maximum concurency allows it.  In openMP
paradigm this is the only way to allow a user to map the proper amount
of HW ressources to a given process (e.g. it allows to pin threads to
cores, and achieve best possible performance).

However, with Threadpool and TBB to some extent, this assumption falls
off. The two latter runtimes usually target improving parallel
execution when a system is oversubscribed. Not when the end-user (the
one starting the process) controls the process execution entirely.

For these two last runtimes, oneDNN should ideally spawn as many task
as possible, while still guarenteeing a minimal granularity for each
task so that it runs with best possible efficiency. This will
typically improve performance in two scenario:
- when hyperthreading is enabled, oneDNN has no way to know how will
  worker threads be mapped to physical cores. If oneDNN spawns only N
  tasks when N cores are available, it will be likely that those tasks
  will compete for the same cores, leaving the other cores idle.
- when a system is oversubscribed, 


Unfortunately implementing such an approach is time consuming and
requires to run a lot of experiments before oneDNN achieves a good
level of performance all around with these new heuristics.

# 3. Proposal

Because rewriting all implementations threading heuristics will be
time consumming and tedious, we propose to expose an API to specify
the maximum concurrency at creation time, as mentioned in section
#how-to-know-the-maximum-concurrency-at-creation-time.

This will be a stop gap solution, and oneDNN developers
can independently work on improving the partitioning heuristics.

## Option 1: global free functions (recommended).

Here we would allow user to modify the maximum concurrency available
to oneDNN by using a global function. This will be enabled only for
threadpool. 

```c++
dnnl_status_t dnnl_threadpool_interop_set_max_concurrency(
        int max_concurrency);

dnnl_status_t dnnl_threadpool_interop_get_max_concurrency(
        int *max_concurrency);
```

The main user of the threadpool threading runtime is Tensorflow.  It
is common in Tensorflow that multiple primitives are created
concurrently and issued in separate threadpools. As such, the oneDNN
max concurrency setting has to be thread local. Otherwise it would
require the oneDNN integration in Tensorflow to use mutex and to
consider oneDNN primitive creation as critical regions .
  
## Option 2: add the entire threadpool in engine

An alternative would be to add a threadpool inside the engine.  This
will allow the primitive creation step to know everything there is to
know about the threadpool that will be used for execution.

Currently, the threadpool is passed to stream creation.
If we pass the threadpool at engine creation, this will have 2 side effects:
- it will have to be compatible with the one that is currently passed
  in stream. In order to avoid overheads of checking that both engine
  and stream threadpools are compatible, we would likely have to drop
  make_stream with threadpool. This is a breaking change and would
  happen only in v3. As a consequence, this will delay adoption of
  this solution by the Tensorflow team (it will require them to fully
  migrate to oneDNN v3).
- it will have a different workflow that with other backends, where we
  always encapsulate the execution stream in the stream object, not in
  the engine.

