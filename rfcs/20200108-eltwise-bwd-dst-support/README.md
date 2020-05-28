# Proposal for supporting DNNL_ARG_DST when computing eltwise backward

## Introduction

### Motivation

- TensorFlow and PyTorch has graph composer which is clever enough to instruct
  to use *the result* of forward computation, in terms of DNNL it's destination,
  when computing backward eltwise. DNNL doesn't provide an ability to specify
  destination memory, which is a pain-point for mentioned FWKs.

- The reason they ask it for is it's very hard for them to modify graph composer
  internal logic and to force it to prepare graph/IR using source data, but not
  destination one. Working around by using inverse operation to get the
  source data from correspondent destination results in high absolute errors
  leading to accuracy drops on a top level.

- Besides accuracy some little performance gain is expected due to reduced
  amount of operations would be required to compute the same answer.

## Proposal

Thanks Evarist for most ideas and insights proposed below.

### Option 1: add a new set of algorithm kinds.

C API:
~~~c
typedef enum {
    ...
    dnnl_eltwise_relu_use_dst_for_bwd = 0x30, // alpha >= 0 only
    dnnl_eltwise_tanh_use_dst_for_bwd = 0x40, // (1 + tanh(x)) * (1 - tanh(x))
    dnnl_eltwise_elu_use_dst_for_bwd = 0x50, // elu(x) + alpha, alpha >= 0 only
    dnnl_eltwise_sqrt_use_dst_for_bwd = 0x70, // 1 / (2 * sqrt(x))
    dnnl_eltwise_logistic_use_dst_for_bwd = 0x80, // logistic(x) * (1 - logistic(x))
    dnnl_eltwise_exp_use_dst_for_bwd = 0x90, // exp(x)
    ...
} dnnl_alg_kind_t;
~~~

In such case user commits to pass memory associated with DNNL_ARG_DST on
backward path. By putting in header file only alg kinds where dst is possible to
support, no weird situations occur when user wants to use destination data but
algorithm simply doesn't support such behavior since maths limitations [1].

Forward path will support and work exactly the same as for alg kinds without
"_use_dst_for_bwd" suffix.

This approach is relatively cheap to implement, it preserves ABI, API,
default/current behavior, doesn't change programming model, has straightforward
documentation.

There's a nice bonus having these new alg kinds. They'll help for backward
operations which have fused eltwise post_ops. Now it's required to stash an
intermediate result of primary operation to pass it for eltwise backward.
Using destination data instead helps removing this stash.

### Option 2: introduce a new attribute.

C API:
~~~c
dnnl_status_t DNNL_API dnnl_primitive_attr_set_eltwise_use_dst_for_bwd(
            dnnl_primitive_attr_t attr, const bool use_dst);
~~~

In this scenario user can specify if using dst is desired. This causes issue [1]
mentioned above, due to attributes are passed after op_desc is created and
there's no programmable way to inform user about inconsistent configuration.
The only possible thing is implementation iterator spits unimplemented status,
which may be confusing for an end-user.

This leads to advanced documentation, which should clarify the behavior in all
possible scenarios of supported and unsupported algorithms.

### Option 3: new primitive.

This option is identical to what happened with logsoftmax. New primitive kind
which makes user to pass dst as an input, shortened list of supported
algorithms.

Likely the cleanest solution as no confusion happens what to pass in execute,
but costs much and introduces new API.

### Option 4: implicit dispatch based on hint.

This option suggests using hint as a decision maker. If passed (always true for
C++ API), set pd state to true if algorithm supports dst as input.

Breaks default behavior on user side. Changes programming model as flag should
be queried for a value and passing src/dst is based on a flag value.

### Option 5: extend existing eltwise_desc_t.

It's a very-very straightforward way to implement a support for such a feature
demanding from a user whether there is a desire to use destination or not.

Unfortunately, it breaks ABI. Also breaks or changes API. Plus it must be
properly documented which algorithms support this flag and which - don't.

### Option 6: pure run-time dispatching.

Introduces nothing new to the API and checks for argument map what was passed.

The main problem is pd cannot be queried for an argument to be passed which
breaks the concept of query.

Inconsistent behavior will have to result in run-time error in execute, rather
than at op_desc or pd creation time.

Additionally will bring some performance burden if backward will start
supporting jit kernel as two versions will have to be created and dispatched at
run-time.

### Comparison of the options

Comparison between options:

| Comparison category                | Option 1         | Option 2      | Option 3      | Option 4      | Option 5      | Option 6      |
|------------------------------------|------------------|---------------|---------------|---------------|---------------|---------------|
| Implementation cost                | *Low-to-Medium*  | Medium        | High          | Medium        | Low-to-Meduim | Meduim        |
| Proper error handling              | *Yes*            | No            | *Yes*         | *Yes*         | No            | No            |
| Preserves ABI                      | *Yes*            | *Yes*         | *Yes*         | *Yes*         | No            | *Yes*         |
| Preserves API                      | *Yes*            | *Yes*         | No            | *Yes*         | No            | *Yes*         |
| Preserves default/current behavior | *Yes*            | *Yes*         | *Yes*         | No            | No            | *Yes*         |
| Changes programming model          | *No*             | *No*          | *No*          | Yes           | Yes           | *No*          |
| Documentation clarity              | *High*           | Medium        | *High*        | Medium        | Medium        | Medium        |

### Recommendation

* The recommendation to pursue option 1 due to minimum number of side effects
  and relatively cheap cost.

### Open questions

### Testing

Well defined for any approach taken.

## Appendices

Arch meeting 01-14-2020 approved to go with Option 1.

