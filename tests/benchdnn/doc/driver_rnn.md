# RNN Driver

## Usage
``` sh
    ./benchdnn --rnn [benchdnn-knobs] [rnn-knobs] [rnn-desc] ...
```

where *rnn-knobs* are:

 - `--prop={FWD_D [default], BWD_DW}` -- mkldnn_prop_kind_t. Refer to the common
            glossary in README.md for details.
 - `--cfg={f32 [default], u8u8u8u8, ...}` -- refer to ``Configurations`` below.
 - `--alg={VANILLA_RNN [default], VANILLA_LSTM, VANILLA_GRU, LBR_GRU}`
            -- RNN algorithm.
 - `--direction={left2right [default], right2left, concat, sum}` -- TBA.
 - `--activation={RELU [default], LOGISTIC, TANH}` -- TBA.
 - `--scaling="scale_str"` -- RNN scaling policy, default `""` (no scaling).
            Refer to knobs_attr.md for details.
 - `--mb=N` -- override minibatch size specified in the problem description.
             When set to `0`, use minibatch size as defined by the individual
             problem descriptor. The default is `0`.

and *rnn-desc* is a problem descriptor. The canonical form is:
```
 lXtXmbX_sicX_slcX_dicX_dlcX_nS
```
Here X is an integer number and S is a string (n stands for name).
The special symbol `_` is ignored, so it may be used as a delimiter.

Description of RNN descriptor symbols: TBA.

There are default values for some entities in case they were not specified:
 - l = 1;
 - t = 1;
 - mb = 2;
 - slc = dlc = dic = sic;


## Precision Configurations

`--cfg` option specifies what data type will be used for a problem. It also
defines the data filling strategy. It is implicit for the integer type
saturation. This option also defines the threshold for computation errors.

The table below shows supported name configurations for this driver:

| states | input | dst_iter  | dst_last_layer | cfg         | notes
|:---    |:---   |:---       |:---            |:---         |:---
| f32    | f32   | f32       | f32            | f32         | TBA
| u8     | u8    | u8        | u8             | u8u8u8u8    | TBA
| u8     | u8    | u8        | f32            | u8u8u8f32   | TBA
| f32    | u8    | f32       | u8             | f32u8f32u8  | TBA
| f32    | u8    | f32       | f32            | f32u8f32f32 | TBA
| f16    | f16   | f16       | f16            | f16         | Only for GPU


## Essence of Testing
TBA.


## Examples. TBA.

Run the set of rnn training from input file with default settings:
``` sh
    ./benchdnn --rnn --batch=inputs/rnn/rnn_training
```

More examples with different driver options can be found at
inputs/rnn/test_rnn_***. Examples with different driver descriptors can be found
at inputs/rnn/rnn_***. More examples with different benchdnn options can be
found at driver_conv.md.
