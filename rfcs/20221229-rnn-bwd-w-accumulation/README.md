Proposal for a RNN backward w.r.t. weights specific mode to not accumulate diff weights
=======================================================================================

## 1. Introduction

## 1.1 OneDNN behavior

The current behavior of RNN for computing diff weights is to accumulate diff into buffer.
The initial idea was to provide a way for the frameworks to split RNN across
time dimension and that required each RNN primitive to accumulate diff weights:

```python
diff_weights = 0.0f
diff_weights += RNN(prop_kind=bwd_w, time=[0,3])
diff_weights += RNN(prop_kind=bwd_w, time=[4,7])
```

An alternative implementation without an accumulation would require extra add
operations:

```python
diff_weights_1 = RNN(prop_kind=bwd_w, time=[0,3])
diff_weights_2 = RNN(prop_kind=bwd_w, time=[4,7])
diff_weights = diff_weights_1 + diff_weights_2
```

## 1.2 Problem statement

In case RNN is not split across time dimension there is no need in
accumulation and `diff_weights` initialization. Since oneDNN semantics for this
operation is defined as an accumulation frameworks have to initialize
`diff_weights` with zeros which results in an overhead:

```python
diff_weights = 0.0f
diff_weights += RNN(prop_kind=bwd_w, time=[0,7])
```


## 2. Proposal (authored by @mgouicem)

The proposal is to remove initialization overhead by *extending* oneDNN semantic
for backward pass of RNN w.r.t. weights:

```python
# Existing behavior
diff_weights += RNN(prop_kind=bwd_w, time=[0,7], flags=None)

# New behavior (new flag)
diff_weights = RNN(prop_kind=bwd_w, time=[0,7], flags=do_not_accumulate)
```

Notes:
- The flag will be ignored in case of forward pass.

## 2.1 API

```cpp
// dnnl_types.h

/// Flags for RNN cell.
typedef enum {
    /// Undefined RNN flags
    dnnl_rnn_flags_undef = 0x0,
    /// Do not add weights gradient to existing diff_weights memory
    dnnl_rnn_flags_diff_weights_overwrite = 0x1
} dnnl_rnn_flags_t;

// dnnl.hpp

/// RNN cell flags.
enum class rnn_flags : unsigned {
    /// Undefined RNN flags
    undef = dnnl_rnn_flags_undef,
    /// Do not add weights gradient to existing diff_weights memory
    diff_weights_overwrite = dnnl_rnn_flags_diff_weights_overwrite
};
```

---

EOD
