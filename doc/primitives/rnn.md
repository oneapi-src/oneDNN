RNN {#dev_guide_rnn}
====================

>
> [API Reference](@ref dnnl_api_rnn)
>

The RNN primitive computes a stack of unrolled recurrent cells, as depicted in
Figure 1. `bias`, `src_iter` and `dst_iter` are optional parameters. If not
provided, `bias` and `src_iter` will default to 0.

@img{unrolled_stack_rnn.jpg,Figure 1: Example of stacked recurrent cells unrolled over the time dimension and executed with the left2right direction. Dashed lines represent optional parameters.,}

The RNN primitive supports four modes for evaluation direction:
-   left2right will process the input data timestamps by increasing order
-   right2left will process the input data timestamps by decreasing order
-   bidirectional_concat will process all the stacked layers from
    left2right and from right2left independently, and will concatenate the
    output in dst_layer over the channel dimension.
-   bidirectional_sum will process all the stacked layers from left2right
    and from right2left independently, and will sum the two outputs to
    dst_layer.

Even though the RNN primitive supports passing a different number of channels
for `src_layer`, `src_iter`, `dst_layer`, and `dst_iter`, we always require the
following conditions in order for the dimension to be consistent:
- \f$channels(dst\_layer) = channels(dst\_iter)\f$,
- when \f$T > 1\f$, \f$channels(src\_iter) = channels(dst\_iter)\f$,
- when \f$L > 1\f$, \f$channels(src\_layer) = channels(dst\_layer)\f$,
- when using the `bidirectional_concat` direction,
 \f$channels(dst\_layer) = 2 * channels(dst\_iter)\f$.

The general formula for the execution of a stack of unrolled recurrent cells
depends on the current iteration of the previous layer (\f$h_{t,l-1}\f$ and
\f$c_{t,l-1}\f$) and the previous iteration of the current layer
(\f$h_{t-1, l}\f$). Here is the exact equation for non-LSTM cells:

\f[
\begin{align}
h_{t, l} = Cell(h_{t, l-1}, h_{t-1, l})
\end{align}
\f]
where \f$t,l\f$ are the indices of the timestamp and the layer of the cell being executed.


And here is the equation for LSTM cells:

\f[ \begin{equation*}
(h_{t, l},c_{t,l}) = Cell(h_{t, l-1}, h_{t-1, l}, c_{t-1,l})
\end{equation*}
\f]
where \f$t,l\f$ are the indices of the timestamp and the layer of the cell being executed.

# Cell Functions

The RNN API provides five cell functions:

-   [Vanilla RNN](#Vanilla-RNN), a single-gate recurrent cell,
-   [LSTM](#LSTM), a four-gate long short-term memory cell,
-   [GRU](#GRU), a three-gate gated recurrent unit cell,
-   [Linear-before-reset GRU](#Linear-before-reset-GRU), a three-gate recurrent
    unit cell with the linear layer before the reset gate.

## Vanilla RNN

A single-gate recurrent cell initialized with
`vanilla_rnn_forward::desc` or `vanilla_rnn_forward::desc` as in the following example.
~~~cpp
    auto vanilla_rnn_desc = vanilla_rnn_forward::desc(
        aprop, activation, direction, src_layer_desc, src_iter_desc,
        weights_layer_desc, weights_iter_desc, bias_desc,
        dst_layer_desc, dst_iter_desc);
~~~

The Vanilla RNN cell supports the ReLU, Tanh and Sigmoid activation
functions. The following equations defines the mathematical operation
performed by the Vanilla RNN cell for the forward pass:

\f[
\begin{align}
a_t &= W \cdot h_{t,l-1} + U \cdot h_{t-1, l} + B \\
h_t &= activation(a_t)
\end{align}
\f]

## LSTM

### LSTM (or Vanilla LSTM)

A four-gate long short-term memory recurrent cell initialized with
`lstm_forward::desc` or `lstm_backward::desc` as in the following example.

~~~cpp
    auto lstm_desc = lstm_forward::desc(
        aprop, direction, src_layer_desc, src_iter_h_desc, src_iter_c_desc,
        weights_layer_desc, weights_iter_desc, bias_desc, dst_layer_desc,
        dst_iter_h_desc, dst_iter_c_desc);
~~~

Note that for all tensors with a dimension depending on the gates number, we
implicitly require the order of these gates to be `i`, `f`, \f$\tilde c\f$, and `o`. The
following equation gives the mathematical description of these gates and output
for the forward pass:

\f[
\begin{align}
i_t &= \sigma(W_i \cdot h_{t,l-1} + U_i \cdot h_{t-1, l} + B_i) \\
f_t &= \sigma(W_f \cdot h_{t,l-1} + U_f \cdot h_{t-1, l} + B_f) \\
\\
\tilde c_t &= tanh(W_{\tilde c} \cdot h_{t,l-1} + U_{\tilde c} \cdot h_{t-1, l} + B_{\tilde c}) \\
c_t &= f_t * c_{t-1} + i_t * \tilde c_t \\
\\
o_t &= \sigma(W_o \cdot h_{t,l-1} + U_o \cdot h_{t-1, l} + B_o) \\
h_t &= tanh(c_t) * o_t
\end{align}
\f]

where \f$W_*\f$ are stored in `weights_layer`, \f$U_*\f$ are stored in
`weights_iter` and \f$B_*\f$ are stored in `bias`.

@note
In order for the dimensions to be consistent, we require
\f$channels(src\_iter\_c) = channels(dst\_iter\_c) =
channels(dst\_iter)\f$.

### LSTM with Peephole

A four-gate long short-term memory recurrent cell with peephole initialized with
`lstm_forward::desc` or `lstm_backward::desc` as in the following example.

~~~cpp
    auto lstm_desc = lstm_forward::desc(
        aprop, direction, src_layer_desc, src_iter_h_desc, src_iter_c_desc,
        weights_layer_desc, weights_iter_desc, weights_peephole_desc,
        bias_desc, dst_layer_desc, dst_iter_h_desc, dst_iter_c_desc);
~~~

Similarly to vanilla LSTM tensors with a dimension depending on the gates
number, we implicitly require the order of these gates to be `i`, `f`,
\f$\tilde c\f$, and `o`. For peephole weights, the gates order is `i`, `f`,
`o`. The following equation gives the mathematical description of these gates
and output for the forward pass:

\f[
\begin{align}
i_t &= \sigma(W_i \cdot h_{t,l-1} + U_i \cdot h_{t-1, l} + P_i \cdot c_{t-1} + B_i) \\
f_t &= \sigma(W_f \cdot h_{t,l-1} + U_f \cdot h_{t-1, l} + P_f \cdot c_{t-1} + B_f) \\
\\
\tilde c_t &= tanh(W_{\tilde c} \cdot h_{t,l-1} + U_{\tilde c} \cdot h_{t-1, l} + B_{\tilde c}) \\
c_t &= f_t * c_{t-1} + i_t * \tilde c_t \\
\\
o_t &= \sigma(W_o \cdot h_{t,l-1} + U_o \cdot h_{t-1, l} + P_o \cdot c_t + B_o) \\
h_t &= tanh(c_t) * o_t
\end{align}
\f]

where \f$P_*\f$ are stored in `weights_peephole`, and the other parameters are
the same as in vanilla LSTM.

@note
If the `weights_peephole_desc` passed to the operation descriptor constructor
is a zero memory desciptor, the primitive will behave the same as in LSTM
without peephole.

## GRU

A three-gate gated recurrent unit cell, initialized with
`gru_forward::desc` or `gru_backward::desc` as in the following example.
~~~cpp
    auto gru_desc = gru_forward::desc(
        aprop, direction, src_layer_desc, src_iter_desc,
        weights_layer_desc, weights_iter_desc, bias_desc,
        dst_layer_desc, dst_iter_desc);
~~~

Note that for all tensors with a dimension depending on the gates number, we
implicitly require the order of these gates to be `u`, `r`, and `o`. The
following equation gives the mathematical definition of these gates.

\f[

\begin{align}
u_t &= \sigma(W_u \cdot h_{t,l-1} + U_u \cdot h_{t-1, l} + B_u) \\
r_t &= \sigma(W_r \cdot h_{t,l-1} + U_r \cdot h_{t-1, l} + B_r) \\
o_t &= tanh(W_o \cdot h_{t,l-1} + U_o \cdot (r_t * h_{t-1, l}) + B_o) \\
h_t &= u_t * h_{t-1, l} + (1 - u_t) * o_t
\end{align}

\f]

where \f$W_*\f$ are in `weights_layer`, \f$U_*\f$ are in
`weights_iter`, and \f$B_*\f$ are stored in `bias`.

@note If you need to replace u_t by (1-u_t) when computing h_t, you can
achieve this by multiplying \f$W_u\f$, \f$U_u\f$ and \f$B_u\f$ by \f$-1\f$.
This is possible as \f$u_t = \sigma(W_u \cdot h_{t,l-1} + U_u \cdot h_{t-1, l}
+ B_u)\f$, and \f$1 – \sigma(a) = \sigma(-a)\f$.


## Linear-Before-Reset GRU

A three-gate gated recurrent unit cell with linear layer applied before the
reset gate, initialized with or  as in the following example.
~~~cpp
    auto lbr_gru_desc = lbr_gru_forward::desc(
        aprop, direction, src_layer_desc, src_iter_desc,
        weights_layer_desc, weights_iter_desc, bias_desc,
        dst_layer_desc, dst_iter_desc);
~~~


The following equation describes the mathematical behavior of the
Linear-Before-Reset GRU cell.

\f[

\begin{align}
u_t &= \sigma(W_u \cdot h_{t,l-1} + U_u \cdot h_{t-1, l} + B_u) \\
r_t &= \sigma(W_r \cdot h_{t,l-1} + U_r \cdot h_{t-1, l} + B_r) \\
o_t &= tanh(W_o \cdot h_{t,l-1} + r_t *(U_o \cdot h_{t-1, l} + B_{u'}) + B_o) \\
h_t &= u_t * h_{t-1, l} + (1 - u_t) * o_t
\end{align}

\f]

Note that for all tensors with a dimension depending on the gates number, except
the bias, we implicitly require the order of these gates to be `u`, `r`, and
`o`. For the `bias` tensor, we implicitly require the order of the gates to be
`u`, `r`, `o`, and `u'`.

@note If you need to replace u_t by (1-u_t) when computing h_t, you can
achieve this by multiplying \f$W_u\f$, \f$U_u\f$ and \f$B_u\f$ by \f$-1\f$.
This is possible as \f$u_t = \sigma(W_u \cdot h_{t,l-1} + U_u \cdot h_{t-1, l}
+ B_u)\f$, and \f$1 – \sigma(a) = \sigma(-a)\f$.

# Data Types

The following table lists the combination of data types supported by the RNN
primitive for each input and output memory object.

 Propagation                | Cell Function | Input data | Recurrent data | Weights | Bias | Output Data
--------------------------- | ------------- | ---------- | -------------- | ------- | ---- | ------------
 Forward / Backward         |  All          | f32        | f32            | f32     | f32  | f32
 Forward                    |  All          | f16        | f16            | f16     | f16  | f16
 Forward inference          |  Vanilla LSTM | u8         | u8             | s8      | f32  | u8, f32

@warning
    There might be hardware and/or implementation specific restrictions.
    Check [Implementation Limitations](@ref dg_rnn_impl_limits) section below.

# Data and Weights Formats

In the DNNL programming model, the RNN primitive is one of a few that
support the placeholder memory format memory::format::any (shortened
to `any` from now on) and can define data and weight memory objects format based
on the primitive parameters.

The following table summarizes the data layouts supported by the RNN
primitive.

 Input/Output Data | Recurrent Data | Layer and Iteration Weights | Peephole Weights and Bias
------------------ | -------------- | --------------------------- | ------------------------
 any               | any            | any                         | ldgo
 ntc, tnc          | ldnc           | ldigo, ldgoi                | ldgo

While an RNN primitive can be created with memory formats specified
explicitly, the performance is likely to be sub-optimal.  When using `any` it
is necessary to first create an RNN primitive descriptor and then query it for
the actual data and weight memory objects formats.

@note
The RNN primitive supports padded tensors and views. So even if
two memory descriptors share the same data layout, they might still be
different.


# Considerations for Training

When using the RNN API for training, the forward pass should use the
`forward_training` propagation kind, and a workspace should be passed to
both the forward pass and the backward pass. Note that after executing the
backward pass, the workspace is no more valid and should be populated
once again by another forward pass.

@anchor dg_rnn_impl_limits
# Implementation Limitations

1. Refer to @ref dev_guide_data_types for limitations related to data types
   support.

2. **GPU**
    - No support for GRU
    - No support for Peephole LSTM
