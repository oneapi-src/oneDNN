# Introducing Max Partition Policy to oneDNN Graph API

## Motivation

oneDNN Graph Compiler is currently a experimental backend for oneDNN Graph API.
It provides
optimized implementations for complex computational graphs including multi-head
attention (MHA), multi-layer perceptron (MLP), and convolution residual blocks
over typical data types for both inference and training.

While the predefined patterns meet some user requirements, there is a growing
interest in customized ops and patterns, which the current fixed patterns
cannot fully accommodate. For instance, users may want to build RMSNorm with
small ops and expect the library to provide optimizations for it.
This is a frequent request for newly introduced activation or normalization.

To address this need and provide users with more flexible and efficient options,
oneDNN Graph Compiler is seeking to incorporate a max partition policy into oneDNN
Graph API, enabling direct programming and unlocking further performance optimizations.

## Current Status

In oneDNN Graph API programming model, a computation graph is passed to library
and then optimized partitions are returned by the library by
`graph.get_partitions()` API.

```c
dnnl_status_t dnnl_graph_graph_filter(
        dnnl_graph_graph_t graph, dnnl_graph_partition_policy_t policy);
```

```cpp
class graph {
    ...
    std::vector<partition> get_partitions(
            partition::policy policy = partition::policy::fusion);
    ...
};
```

A partition policy is used in the API to control granularities
of optimizations to be applied when partitioning.
Currently, oneDNN Graph API supports two partition policies:

```c
typedef enum {
    dnnl_graph_partition_policy_fusion = 1,
    dnnl_graph_partition_policy_debug = 2,
} dnnl_graph_partition_policy_t;
```

```cpp
class partition {
    ...
    enum class policy {
        fusion = dnnl_graph_partition_policy_fusion,
        debug = dnnl_graph_partition_policy_debug,
    };
    ...
};
```

- Fusion policy (default partition policy): fusion policy returns partitions
  with [pre-defined fusion patterns](https://oneapi-src.github.io/oneDNN/dev_guide_graph_fusion_patterns.html),
  e.g. Convolution followed by a chain of post-ops, MHA, MLP, etc.
- Debug policy: debug policy returns partitions with single operation in each partition.
  This policy doesn't apply any fusions, which is useful when
  user wants to debug or compare performance with fusion policy.

## Proposal

As mentioned in previous sessions, fusion policy relies on predefined fusion
patterns to do partitioning, which may miss some optimization opportunities
for customized ops and patterns that have not been predefined.
To address this, max policy is proposed.

The expected behaviors when using max policy:

- Max policy relies on predefined ops list to do partitioning. Connected ops
  that meet the conditions of predefined ops list (and predefined rules)
  will be selected into one partition.
- Max policy uses a smaller granularity (predefined ops) to do partitioning
  compared to fusion policy, which relies on predefined patterns.
  This results in larger partitions and is the reason behind the name `max`.
- If not specified, the default partition policy remains to be fusion policy.
- The predefined ops list (and predefined rules) for max policy is backend specific.

The API to expose max policy to users is similar to fusion and debug policies.

```c
typedef enum {
    dnnl_graph_partition_policy_max = 0, // the new max policy
    dnnl_graph_partition_policy_fusion = 1,
    dnnl_graph_partition_policy_debug = 2,
} dnnl_graph_partition_policy_t;
```

```cpp
class partition {
    ...
    enum class policy {
        max = dnnl_graph_partition_policy_max, // the new max policy
        fusion = dnnl_graph_partition_policy_fusion,
        debug = dnnl_graph_partition_policy_debug,
    };
    ...
};
```

## Validation

- To validate the functionality of max partition policy, corresponding gtests will
  be added.
- To validate the correctness and performance of max partition policy, we will
  utilize the benchdnn graph tool.
  - To control which partition policy to use in `graph.get_partitions()` API,
    new command line parameter or new environment variable needs to be added
    to benchdnn graph.
  - For correctness check, we can utilize the current implementation of the
    reference path to validate the numarical results of max partition policy.
  - For performance check, as currently there's no baseline for the performance of
    max partititon policy, the first round of performance check will be set as
    baseline for later performance check.

## Open questions

1. Compared with fusion and debug policy, max policy gives the library more control
   on partitioning. This on the other hand means users may lose fine-grained control
   of what optimizations they want to get from the library.
1. The performance of max policy may or may not be better than fusion policy. The
   library tries to provide the best performance by model performance heuristics.
   But for untested models, it is possible that the performance of max policy is
   worse than fusion policy.

## Known issues

1. For all backends that don't support max partition policy, they will fallback to
   fusion partition policy implicitly.
   Likewise, for all backends that don't support fusion partition policy, they
   will fallback to debug partition policy implicitly.
   For example, DNNL backend will not support max partition policy in oneDNN v3.3
   release, so when running with a combination of max partition policy and DNNL
   backend, max partition policy will fallback to fusion partition policy.
1. The validation for max partition policy will not be landed in oneDNN v3.3
   release, so the implementation of max partition policy in Graph Compiler
   backend is a experimental feature.

END
