# Dice function

As part of a new click-through rate prediction model [1], Alibaba
introduced a new activation function called Dice.  This function is
gaining adoption with a few customers, who are now asking for
optimizations.

There are two mathematical formulas, one in the paper [1], another in the
DeepRec implementation [2].

```math
\begin{align}
dst[i] &= p(src[i]) \cdot src[i] + (1- p(src[i])) \cdot \alpha[i] \cdot src[i] \\
       &= src[i] \cdot ( \alpha[i] + (1 - \alpha[i]) \cdot p(src[i]) )
\end{align}
```

with $p(src[i]) = sigmoid ((src[i] - E[i]) / \sqrt{V[i] + \epsilon})$.
These two formulas are mathematically equivalent, but not numerically
equivalent when evaluated in floating-point arithmetic.

From the paper [1] as well as the pull-request to DeepRec [2], the
parameters $\alpha[i]$, $E[i]$ and $V[i]$ are vectors. In particular,
$E[i]$, $V[i]$ and $\alpha[i]$ are computed wrt to rightmost logical
dimension of the tensor (similar to layer normalization [3]).

Also note that Dice can be viewed as a generalization of PReLu, as
using $E[i] = V[i] = 0$ yields the same result as PReLu.


## Proposal

The Dice formula cannot currently be computed without using multiple
oneDNN primitive calls and intermediate memory buffers (because
intermediate expressions in the formula involve both `src` and
`p(src)`).


There are three options for API:
- extend PReLU with mean/variance to cover Dice functionality,
- introduce a new primitive for Dice specifically,
- introduce a new trainable eltwise primitive, that takes algorithm as
  parameter (Dice/PReLU only for now).

On the implementation side PReLu and Dice can rely on the same
implementation, as PReLu is a sub-case of Dice with no mean/variance.

Regarding post-op computation, mean and variance are pre-computed and
simply passed to the post-op, so they don't need to be computed during
post-op chain.

Regarding backward computation there was no request to support it for
now.  However, it is worth noting than `E[i]` and `V[i]` are constant and
computed in previous layer, so we don't have to compute them like in
normalization layers.

### Option 1: extend PReLU with statistics.

Because DICE is a generalization of PReLU, we could also just extend
PReLU primitive and post-op with a new `with_stats` parameter.

The new C++ constructor for `prelu_forward` would be
```c++
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                const memory::desc &src_desc, const memory::desc &weight_desc,
                const memory::desc &dst_desc,
                bool with_stats = false,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
```

For post-ops, we would have 2 new entry points

```c++

/// C entrypoint
dnnl_status_t DNNL_API dnnl_post_ops_append_prelu_v2(
        dnnl_post_ops_t post_ops, int mask, bool with_stats);

dnnl_status_t DNNL_API dnnl_post_ops_get_params_prelu_v2(
        const_dnnl_post_ops_t post_ops, int index, int *mask, bool *with_stats);
```

and their corresponding C++ counterparts:

```c++
/// C++ entry
    void append_prelu(int mask, bool with_stats=false);
    void get_params_prelu(int index, int &mask, bool &with_stats);
```

Pros:
- very simple API additions, and no new primitive kind needed
- allows to piggyback on prelu testing

Cons:
- from user perspective, it might not be clear oneDNN has Dice
  support. This will have to be properly documented. Optionally, we
  could also

### Option 2: introduce dice primitive and post-op

For start, we would introduce only the forward version, not backward,
as there was no request for training support as of now.  This can
easily be extended at a later time though.

```c++
struct dice_forward : public primitive {
    struct primitive_desc : public dnnl::primitive_desc {
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                const memory::desc &src_desc, const memory::desc &weight_desc,
                const memory::desc &mean, const memory::desc &variance,
                const memory::desc &dst_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
	}
}

```


For post-op, it will be very similar to PReLU, as all parameters
(weights and stats) share the same layout

```c++
    void append_dice(int mask) {
        error::wrap_c_api(dnnl_post_ops_append_dice(get(), mask),
                "could not append a dice post-op");
    }
    void get_params_dice(int index, int &mask) const {
        error::wrap_c_api(dnnl_post_ops_get_params_prelu(get(), index, &mask),
                "could not get parameters of a dice post-op");
    }
```

Pros:
- Clear API 

Cons:
- not scalable approach. Next trainable eltwise will require new
  primitive kind, and new testing driver in benchdnn.


### Option 3: introduce parametrized_eltwise

```c++
struct parametrized_eltwise_forward : public primitive {
    struct primitive_desc : public dnnl::primitive_desc {
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &weight_desc,
                const memory::desc &dst_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
	}
}


struct parametrized_eltwise_backward : public primitive {
    struct primitive_desc : public dnnl::primitive_desc {
        primitive_desc(const engine &aengine, prop_kind aprop_kind,
                algorithm aalgorithm, const memory::desc &src_desc,
                const memory::desc &weight_desc,
                const memory::desc &diff_src_desc,
                const memory::desc &diff_weight_desc,
                const memory::desc &diff_dst_desc,
                const primitive_attr &attr = default_attr(),
                bool allow_empty = false)
	}
}

```

In the above, all parameters will share the same descriptor. For
example:
- For prelu there is only a single set of weights, so it is fully
  described by weight_desc,
- For Dice `mean`, `var` and `alpha` will all have the same shape,
  described by weight_desc. This is consistent with Alibaba dice
  implementation

This new primitive will take execution arguments depending on the
algorithm.
- for PReLU: `DNNL_ARG_SRC`, `DNNL_ARG_DST`, `DNNL_ARG_WEIGHTS`
- for Dice: `DNNL_ARG_SRC`, `DNNL_ARG_DST`, `DNNL_ARG_WEIGHTS`,
  `DNNL_ARG_MEAN`, `DNNL_ARG_VARIANCE`

Pros:
- more scalable long term: simpler validation with a single benchdnn
  driver, no new primitive for each algorithm.

Cons:
- constrains the dimensions of extra inputs to those of
  src/dst/weights descriptors.



## Implementation details

Whatever API option we adop, we can enable Dice by extending internal
implementation of PReLu to handle statistics.

The recommendation here would be to consider Dice a generalization of
PReLU, in which case all impls will keep the PReLU naming, PReLU
internal operation descriptor will be extended with a `with_stats`
flag. For post-op, the same applies, we can add a `with_stats` flag to
internal prelu post-op attribute.

Regarding testing, we recommend benchdnn extension to match API: if
new primitive, a new driver would be added, if PReLU extension, we
will add the flag to the PReLU driver.

## Recommendation
The recommendation is to extend PReLU to handle Dice (Option 1) as it
is a simpler API and tests extension.  If a new primitive is prefered,
I would advocate for a new parametrized_eltwise (option 3).


## References
- [1]: Original paper https://arxiv.org/pdf/1706.06978.pdf
- [2]: Alibaba [pull request](https://github.com/DeepRec-AI/DeepRec/pull/581/) to DeepRec.
- [3]: [oneDNN layer normalization documentation](https://oneapi-src.github.io/oneDNN/dev_guide_layer_normalization.html)
