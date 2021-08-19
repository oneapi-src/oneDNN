# Benchdnn performance measurement issues.

## Problem Statement.
There are talks around and inside oneDNN team about the need to improve the way
performance validation/benchmarking of E2E models or their parts can be done.

Currently, benchdnn provides the only way to measure performance - single
problem is executed on same memory until certain "time" conditions are reached.
This allows to know a problem performance in "vacuum" environment - unrealistic
conditions for end users. It still shows how efficient the kernel is, this is
still an important part, but final performance values are not limited to kernel
efficiency in "vacuum" environment only. Each next generation of hardware brings
more compute elements and ususally do not improve a memory component which means
that more and more problems become memory-bound rather than compute bound.
Final performance users may see depends on many different factors.

The problem of only existing approach that the result should be somehow scaled
down to be closer to the "reality" but it is hard to say how much and how to
estimate this ratio. Also certain performance problems cannot be found with such
methodology. I.e., cores trash-trafficking effect due to different blocking
strategies in adjacent primitives, or data eviction from caches due to different
sets of weights, while a single problem fits well. It may also happen that HW
features may change behavior in time. For example, compute frequency changes
triggered by using some compute instructions (Intel AVX-512, Intel AMX, or
DPAS).

All these or other problems are hard or impossible to observe with existing
benchdnn performance measuring toolset. That is why, developers or users who
report issues have to use framework integration of oneDNN and submit trackers
using this heavy-weighted instrument. There are several down sides of this
approach:
1) It requires frameworks infrastructure to be enabled on developer side.
   Running different scripts is not that bad as trying to modify codes or other
   environment around. It usually requires consultation from other teams or
   reporter.
2) To validate something new, especially API, it requires integration to be
   implemented on framework side. It takes some time to make it done, which
   postpones validation and may result in milestones misses when it comes to
   new hardware enabling, delivering certain level of performance, etc.
3) When it comes to performance measurements and analysis, it may be difficult
   to distinguish a problem on framework side (integration code may have issues)
   from the problem on the library side. In addition, there are certain
   overheads that should be accounted when doing analysis.

All these issues create a need to extend benchdnn performance measurements
capabilities.

## Proposal
There are two general approaches that could be taken in order to improve a
toolset:
* "cold cache" mechanism. When executing a hot loop it would evict data from
  cache(s) after each execution making new iteration to move data from RAM or
  higher level cache(s).

  Open questions:
  - Not clear which memories to flush if there are two or more input tensors.
  - Not clear what should be a level of eviction.
  - What UI to provide to handle cons with enough level of flexibility?

* Chain several primitives to execute them consecutive. User passes
  benchdnn-style problems they want to run consecutively, new mode will parse
  problems, pile up primitives and correspondent memories and then run a full
  chain in a loop instead of a single problem passing data from layer to layer.

It may be a good idea to have both approaches implemented to have a variation
for performance benchmarking since they are not intersecting in their meaning
and value to deliver.

EOD.
