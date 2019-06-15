## benchdnn

**benchdnn** is a standalone correctness and performance benchmark for
[Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN)](
/intel/mkl-dnn).

The purpose of the benchmark is extended and robust correctness verification of
the primitives provided by Intel MKL-DNN.
**benchdnn** itself is a harness for different primitive-specific drivers.
So far it supports and uses the following drivers:
* [batch normalization](tests/benchdnn/doc/driver_bnorm.md)
* [convolution](tests/benchdnn/doc/driver_conv.md)
* [deconvolution](tests/benchdnn/doc/driver_conv.md)
* [inner product](tests/benchdnn/doc/driver_ip.md)
* [pooling](tests/benchdnn/doc/driver_pool.md)
* [reorder](tests/benchdnn/doc/driver_reorder.md)
* [recurrent neural network (RNN)](tests/benchdnn/doc/driver_rnn.md)
* [shuffle](tests/benchdnn/doc/driver_shuffle.md)
* [softmax](tests/benchdnn/doc/driver_softmax.md)
* [sum](tests/benchdnn/doc/driver_sum.md)

## License
**benchdnn** is licensed under
[Apache License Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

## Harness Usage
``` sh
    ./benchdnn --DRIVER [--engine=ENGINE_KIND] [--mode=MODE] [--reset] \
               [--max-ms-per-prb=TIME-IN-MS] [--fix-times-per-prb=N] \
               [-vN|--verbose=N] [--skip-impl=SKIP_IMPL] [--allow-unimpl=BOOL] \
               [--perf-template=PERF_TEMPLATE] \
               [DRIVER-OPTS] PROBLEM-DESCRIPTION [--batch=FILE]
```

where:

 - `--DRIVER` -- is either `bnorm`, `conv` [default], `deconv`, `ip`, `pool`,
            `reorder`, `rnn`, `shuffle`, `softmax`, or `sum`.
 - `--engine=ENGINE_KIND` -- specifies the engine kind to use for the benchmark.
            Can be `cpu` [default] or `gpu`.
 - `--mode=MODE` -- string that contains flags for benchmark mode.
            `C`, `c` for correctness [default], `P`, `p` for performance.
 - `--reset` -- reset all the parameters set previously to the default.
 - `--max-ms-per-prb=TIME-IN-MS` -- time spent per problem in milliseconds.
            Available range [1e2, 60e3]. Default is `3e3`.
 - `--fix-times-per-prb=N` -- number of iterations spent per problem, N must be
            non-negative. Default is `0` (not applied, time criterion is used).
 - `-vN, --verbose=N` -- verbose level; use for printing additional information.
            Default is `0`.
 - `--skip-impl="str1[:str2]..."` -- skip a specific implementation
            (see mkldnn_query_impl_info_str), default `""`.
 - `--allow-unimpl=true|false` -- do not treat unimplemented configuration
            as an error. Default is `false`.
 - `--perf-template={def [default], csv, CUSTOM_TEMPLATE}` -- A template to
            provide the output for a performance run. Refer to
            knobs_perf_report.md for detailed information.
 - `DRIVER-OPTS` -- each driver has a customized list of options. Refer to
            the corresponding driver_DRIVER.md for detailed information.
 - `PROBLEM-DESCRIPTION` -- each driver requires a specific problem format.
            Refer to the corresponding driver_DRIVER.md for detailed
            information.
 - `--batch=file` -- use options from the given file, can handle any options.

Returns `0` on success (all tests passed) or non-zero in case of any error.

## Common Glossary

|Abbreviation   | Description
|:---           |:---
| src           | Source image (input image for forward convolution)
| wei           | Weights (aka filter)
| bia           | Bias
| dst           | Destination image (output image for forward convolution)
| acc           | Accumulation (typically in terms of data type)
| ic, oc        | Input/Output channels (aka feature maps)
| id, ih, iw    | Input depth, height and width
| od, oh, ow    | Output depth, height and width
| kd, kh, kw    | Kernel (filter, weights) depth, height and width
| sd, sh, sw    | Convolution stride over depth, height and width
| pd, ph, pw    | Convolution front, top and left padding
| mb            | Minibatch (amount of images processed at once)
| g             | Groups (a way to reduce the amount of computations, see Alexnet topology)

|Prop kind:     |
| FWD_B         | mkldnn_forward_training w/ bias
| FWD_D         | mkldnn_forward_training w/o bias
| FWD_I         | mkldnn_forward_inference
| BWD_D         | mkldnn_backward_data
| BWD_W         | mkldnn_backward_weights
| BWD_DW        | mkldnn_backward_data + mkldnn_backward_weights
| BWD_WB        | mkldnn_backward_weights + mkldnn_backward_bias

|Data_type/cfg: |
| f32           | standard float
| s32           | standard int
| s8            | standard char
| u8            | standard unsigned char
| f16           | 2-byte float (5 bits exp, 10 bits mantissa, 1 bit sign)
| bf16          | 2-byte float (8 bits exp,  7 bits mantissa, 1 bit sign)

|Format tags:   | Physical disposition of elements in memory
| Plain:        |
|  abcd         | Standard de-facto for training in CNN (aka nchw).
|  acdb         | Standard de-facto for int8 inference in CNN (aka nhwc).
| Blocked:      |
|  aBcd8b       | Internal blocked format for AVX2 systems and below.
|  aBcd16b      | Internal blocked format for AVX512VL systems and above.
| ...           | and some others...
| Special:      |
|  any          | mkldnn_format_tag_any. Let the library decide, which layout should be used.
|  undef        | mkldnn_format_tag_undef. Make a driver omit dst, letting the library to deduce it.


## Issues and Contributions

We welcome community contributions to **benchdnn** as well as to Intel MKL-DNN.
If you have any ideas or issues, please submit an issue or pull request. For
clarity, please include ''benchdnn: '' in the title.


## Inspiration

This work is inspired by [benchFFT](http://www.fftw.org/benchfft/) project
developed by Matteo Frigo and Steven G. Johnson as a benchmark for
Discrete Fourier Transform (DFT) implementations.
