# Introducing Max Partition Policy to oneDNN Graph API

## Motivation

oneDNN Graph Compiler is currently a experimental backend for oneDNN Graph API.
It provides
optimized implementations for complex computational graphs including multi-head
attention (MHA), multi-layer perceptron (MLP), and convolution residual blocks
over typical data types for both inference and training.

To further boost performance and expose more aggressive optimizations to users,
oneDNN Graph Compiler is requesting to add max partition
policy to oneDNN Graph API.

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

The expected behaviors when using max policy:

- If not specified, the default partition policy remains to be fusion policy.
- Max policy returns partitions max in partition size. The criteria of partitioning
  is based on pre-defined ops list and pre-defined rules. Connected ops
  that meet the conditions of pre-defined ops list and pre-defined rules
  will be selected into one partition.
- The pre-defined ops list and pre-defined rules are backend specific.

## Open questions

1. Compared with fusion and debug policy, max policy gives the library more control
   on partitioning. This on the other hand means users may lose fine-grained control
   of what optimizations they want to get from the library.
1. The performance of max policy may or may not be better than fusion policy. The
   library tries to provide the best performance by model performance heuristics.
   But for untested models, it is possible that the performance of max policy is
   worse than fusion policy.

## Known issues

1. DNNL backend will not support max partition policy in oneDNN v3.3 release.
   So when running with a combination of max partition policy and DNNL backend,
   max partition policy will fallback to fusion partition policy.
1. The validation for max partition policy will not be landed in oneDNN v3.3
   release, so the implementation of max partition policy in Graph Compiler
   backend is a experimental feature.

END
