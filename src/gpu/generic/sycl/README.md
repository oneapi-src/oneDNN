# Supported Primitives

## Batch Normalization

The implementation supports both forward and backward directions.

* Supported formats: `NCDHW`, `NDHWC`, `NCHW`, `NHWC`, `NCW`, `NWC`, `NC`

## Eltwise

The implementation supports both forward and backward directions.

* Supported algorithms: `abs`, `clip`, `clip_v2`, `elu`, `exp`, `gelu_erf`,
`gelu_tanh`, `hardsigmoid`, `hardswish`, `linear`, `log`, `logistic`, `mish`,
`pow`, `relu`, `round`, `soft_relu`, `sqrt`, `square`,`swish` and `tanh`
* Supported formats: `NCDHW`, `NDHWC`, `NCHW`, `NHWC`, `NCW`, `NWC`, `NC`, `N`

## LRN

The implementation supports both forward and backward directions.

* Supported formats: `NCDHW`, `NDHWC`, `NCHW`, `NHWC`, `NCW`, `NWC`, `NC`

## Pooling

The implementation supports both forward and backward directions.

* Supported formats: `NCDHW`, `NDHWC`, `NCHW`, `NHWC`, `NCW`, `NWC`

## PReLU

The implementation supports both forward and backward propagations.

* Supported formats: `NCDHW`, `NDHWC`, `NCHW`, `NHWC`, `NCW`, `NWC`, `NC`

* Forward pass supports `f32`, `f16`, `bf16`, `s8` and `u8` data types
* Backward pass supports `f32` and `bf16` data types

## Reorder

* Format support limitations: blocked formats are not supported
* Supported data types: `f32`, `bf16`, `f16`, `s8`, `u8`

## Resampling

The implementation supports both forward and backward directions.

* Supported formats: `NCDHW`, `NDHWC`, `NCHW`, `NHWC`, `NCW`, `NWC`

## Softmax/LogSoftmax

The implementation supports both forward and backward directions.

* Supported formats: `NCDHW`, `NDHWC`, `NCHW`, `NHWC`, `NCW`, `NWC`, `NC`

## Shuffle

The implementation supports both forward and backward propagations.

* Supported formats: `NCDHW`, `NDHWC`, `NCHW`, `NHWC`, `NCW`, `NWC`, `NC`

* Forward pass supports `f32`, `f16`, `bf16` and `s8` data types.
* Backward pass supports `f32` and `bf16` data types.
