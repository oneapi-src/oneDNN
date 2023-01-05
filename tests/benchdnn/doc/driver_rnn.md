# Recurrent Neural Network Driver

## Usage
``` sh
    ./benchdnn --rnn [benchdnn-knobs] [rnn-knobs] [rnn-desc] ...
```

where *rnn-knobs* are:

 - `--prop={FWD_I [default], FWD_D, BWD_DW}` -- dnnl_prop_kind_t.
            Refer to [direction](knobs_dir.md) for details.
 - `--cfg={f32 [default], ...}` -- refer to ``Configurations`` below.
 - `--alg={VANILLA_RNN [default], VANILLA_LSTM, VANILLA_GRU, LBR_GRU,
            VANILLA_AUGRU, LBR_AUGRU}` -- RNN algorithm.
 - `--direction={left2right [default], right2left, concat, sum}` -- RNN
            evaluation direction.
 - `--activation={RELU [default], LOGISTIC, TANH}` -- `VANILLA_RNN` activation
            functions.
 - `--scaling="scale_str"` -- RNN scaling policy, default `""` (no scaling).
 - `--scaling-proj="scale_str"` -- RNN scaling policy for the
            projection weights, default `""` (no scaling).
            Refer to [attributes](knobs_attr.md) for details.
 - `--skip-nonlinear={true, false [default]}` -- specify if transcendental
            activations will be treated as linear. This allows to test longer
            chains avoiding errors coming from non-linear activation functions.
            Especially relevant for int8 computations. For LSTM, GRU and AUGRU
            flows changes internal implementation since there is no external
            control over pre-defined activations in a cell.
 - `--trivial-strides={true, false [default]}` -- specify if input tensors
            should have trivial strides or not. Each tensor stride is the
            product of previous dimensions.
 - `--with-peephole={true, false [default]}` -- LSTM extension. Specify if LSTM
            with peephole should be run.
 - `--with-projection={true, false [default]}` -- LSTM extension. Specify if LSTM
            with projection should be run.
 - `--l=INT` -- override `l` (number of layers) value specified in the problem
            descriptor. When `INT` is set to `0` (the default), use `l` value
            specified in the problem descriptor.
 - `--t=INT` -- override `t` (number of timestamps) value specified in the
            problem descriptor. When `INT` is set to `0` (the default), use `t`
            value specified in the problem descriptor.
 - `--mb=INT` -- override `mb` (minibatch) value specified in the problem
            descriptor. When `INT` is set to `0` (the default), use `mb` value
            specified in the problem descriptor.
 - `--attr-fpmath=STRING` -- fpmath mode primitive attribute. `strict` math mode
            is set by default. Refer to [attributes](knobs_attr.md) for details.
 - `--flags=[|O]` -- RNN flags, default `undef` (no flags); where multiple
            simultaneous flags are supported.
            `O` is dnnl_rnn_flags_diff_weights_overwrite;
            Refer to [RNN primitive](https://oneapi-src.github.io/oneDNN/dev_guide_rnn.html) for details.

and *rnn-desc* is a problem descriptor. The canonical form is:
```
 lXtXmbX_sicX_slcX_dhcX_dicX_nS
```
Here `X` is an integer number and `S` is a string literal without spaces (`n`
stands for name). The special symbol `_` is ignored, so it may be used as a
delimiter for better readability.

Description of RNN descriptor symbols:
 - `l` is the number of layers. The default value is `1`.
 - `t` is the number of timestamps (or the sequence length). The default value
   is `1`.
 - `mb` is the minibatch. The default value is `2`.
 - `sic` is the feature size of `src_iter`. No default value.
 - `slc` is the feature size of `src_layer`. The default value is `sic`.
 - `dhc` is the hidden feature size. The default value is `sic`.
 - `dic` is the feature size of `dst_iter`. The default value is `dhc`. For GRU
   and AUGRU it must be equal to `dhc`.


## Precision Configurations

The `--cfg` option specifies the [data types](knobs_dt.md) to be used for a
problem. It also defines the data filling strategy. It is implicit for the
integer type saturation. This option also defines the threshold for computation
errors.

The table below shows supported name configurations for this driver:

| states iter | states iter_c | input | dst_iter  | dst_last_layer | bias | cfg                    | notes
|:---         |:---           |:---   |:---       |:---            |:---  |:---                    |:---
| f32         | f32           | f32   | f32       | f32            | f32  | f32                    | TBA
| bf16        | bf16          | bf16  | bf16      | bf16           | bf16 | bf16                   | TBA
| bf16        | bf16          | bf16  | bf16      | bf16           | f32  | bf16f32                | TBA
| bf16        | f32           | bf16  | bf16      | bf16           | f32  | bf16f32bf16bf16bf16f32 | TBA
| u8          | f32           | u8    | u8        | u8             | f32  | u8u8u8u8               | TBA
| u8          | f32           | u8    | u8        | f32            | f32  | u8u8u8f32              | TBA
| f32         | f32           | u8    | f32       | u8             | f32  | f32u8f32u8             | TBA
| f32         | f32           | u8    | f32       | f32            | f32  | f32u8f32f32            | TBA
| f16         | f16           | f16   | f16       | f16            | f16  | f16                    | Only for GPU


## Essence of Testing
TBA.


## Examples. TBA.

Run the set of rnn training from an input file with the default settings:
``` sh
    ./benchdnn --rnn --batch=inputs/rnn/shapes_training
```

More examples with different driver options can be found at inputs/rnn/test_\*.
Examples with different problem descriptors can be found at
inputs/rnn/shapes_\*. More examples with different benchdnn common options can
be found at driver_conv.md.
