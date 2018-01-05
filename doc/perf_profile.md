# Performance profiling

It is often useful to collect information about how much of an application run
time is spent executing Intel(R) MKL-DNN primitives and which of those take
the most time. One of the popular methods to do this is to use profilers like
Linux\* perf or Intel(R) VTune(tm) Amplifier. Currently, Intel MKL-DNN has very
limited support for these tools since it does not annotate code generated at
run-time and thus the profiles cannot properly attribute it. However, Intel
MKL-DNN implements another feature called _verbose mode_ that allows tracing
execution of Intel MKL-DNN primitives and collection of basic statistics like
execution time and primitive parameters.

## Verbose mode

To enable Intel MKL-DNN verbose mode, set `MKLDNN_VERBOSE` environment variable
to `1` (to dump only execution time) or `2` (to dump both execution and
creation time). For example:

```
    $ export MKLDNN_VERBOSE=1
    $ ./benchdnn --conv ic16ih7oc16oh7kh5ph2n"wip"
```

This will produce the following output (the line break was added to fit into
the page width):

```
    mkldnn_verbose,exec,reorder,undef,in:f32_nchw out:f32_nChw8c,num:1,2x16x7x7,0.484863
    mkldnn_verbose,exec,reorder,undef,in:f32_goihw out:f32_gOIhw8i8o,num:1,1x16x16x5x5,0.494141
    mkldnn_verbose,exec,reorder,undef,in:f32_nchw out:f32_nChw8c,num:1,2x16x7x7,0.478027
    mkldnn_verbose,exec,reorder,undef,in:f32_x out:f32_x,num:1,16,0.219971
    mkldnn_verbose,exec,convolution,forward_inference,fsrc:nChw8c fwei:gOIhw8i8o fbia:x \
        fdst:nChw8c,alg:convolution_direct,mb2_g1ic16oc16_ih7oh7kh5sh1dh0ph2_iw7ow7kw5sw1dw0pw2,0.0170898
    mkldnn_verbose,exec,reorder,undef,in:f32_nChw8c out:f32_nchw,num:1,2x16x7x7,0.488037
    mkldnn_verbose,exec,reorder,undef,in:f32_nChw8c out:f32_nchw,num:1,2x16x7x7,0.00512695
    0:PASSED __REPRO: ic16ih7oc16oh7kh5ph2nwip
    tests:1 passed:1 skipped:0 mistrusted:0 unimplemented:0 failed:0
```

Each line with verbose information is formatted as a comma-separated list
containing:
- `mkldnn_verbose`
- `stage`, e.g. `create` or `exec`
- `primitive-kind`, e.g. `convolution`, `reorder`, `sum`, ...
- propagation-kind, e.g. `forward_training`
- input/output data info, e.g. data type and data format
- auxiliary information, e.g. algorithm or number of input
- problem description
    - for convolution the problem description is dumped in benchdnn friendly format
    - for reorder, sum, and concat problem description is simply logical dims
    - for other primitives the problem description is similar to convolution one
- execution time in milliseconds

To get more information about verbose report format please refer to the
`verbose_templ()` function in the
[src/common/verbose.hpp](https://github.com/01org/mkl-dnn/blob/master/src/common/verbose.hpp)
file.

---
**NOTE**
The format is subject to change

---


---
**WARNING**
Verbose mode has non-negligible performance impact especially if the output
rate is high.

---

[Legal information](legal_information.md)
