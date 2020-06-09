# Introducing logsigmoid operation

## Introduction

Request comes from DGP graph lib community https://www.dgl.ai. Logsigmoid is
used in AWS embedding models https://bit.ly/30kWI1f. Currently logsigmoid can be
implemented with oneDNN by using two eltwise
(https://oneapi-src.github.io/oneDNN/dev_guide_eltwise.html) operations
logistic and log. Unfortunately current eltwise ops are not fusable.
Logistic loads data from memory, make computations, stores data and next log
loads logistic output from memory and stores data (2x loads and 2x stores). While
it could be possibly 1x load and 1x store if these operations were combined.
Eltwise is memory bound primitive, so additional loads and store has performance
impact.

Naming convention taken from: https://oneapi-src.github.io/oneDNN/dev_guide_eltwise.html
FWD derivative calculated using https://www.wolframalpha.com/input/?i=d%2Fdx%28log%281%2F%281%2Bexp%28-x%29%29%29%29

Formulas:
- forward_logsigmoid - d = log(1/(1+exp(-x)));
- backward_logsigmoid - ds = 1/(exp(s)+ 1) * dd;
- backward_logsigmoid_dst - not possible;

For numerical stability suggested formulas:
- forward_logsigmoid - d = -soft_relu(-x)
- backward_logsigmoid - ds = 1/(exp(s)+ 1) * dd;
- backward_logsigmoid_dst - not possible;

## Proposal

**Option 1 Extend exisiting eltwise with new operation:**

``` cpp
typedef enum {
    ...
    dnnl_eltwise_logsigmoid = number;
    ...
} dnnl_alg_kind_t;
```

Pros:
- the simplest possible solution;
- reuse existing implementation internally;

Cons:
- flexibility: extending op list each time when exisiting can be fused seems to
be poor general solution;

**Option 2 Add support for eltwise post-ops in eltwise primitive**

``` cpp
    auto eltwise_d = eltwise_forward::desc(prop_kind::forward_training,
            algorithm::dnnl_eltwise_logistic, src_md, 0.f, 0.f);

    post_ops eltwise_ops;
    eltwise_ops.append_eltwise(scale, algorithm::eltwise_log, 0.f, 0.f);
    primitive_attr eltwise_attr;
    eltwise_attr.set_post_ops(eltwise_ops);

    auto eltwise_pd = eltwise_forward::primitive_desc(eltwise_d, eltwise_attr, engine);
```

Pros:
- reuse existing implementation internally;
- flexibility - can be used each time when there will be need for fuse existing
ops;

Cons:
- more complicated in implementation than option 1;
- problem with reusing data from forward passed in backward should be addressed
with generic solution;

I recommend implementing option 1, because of simplicity. Option 2 has
problems with general backward applicability, which should be solved in another
RFC if it will be needed in more than one case.
