# Verbose log converter

Verbose log converter is a tool that allows to convert [oneDNN
verbose](https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html)
output to various outputs (input files for benchdnn and execution
statistics breakdown at this time). The tool can be extended to
produce other types of output by adding generators.

## Requirements
 - Python 3.6

## Compatibility
The script is compatible with the specific oneDNN version with which it is distributed.
Compatibility with other oneDNN versions is not guaranteed.
To get an appropriate version of the script:
 - Identify `DNNL_VERSION_HASH` located in `include/oneapi/dnnl/dnnl_config.h`.
 - Download the script from oneDNN repository with that particular hash.

## Usage
### Option 1: call from command line
``` sh
python3 verbose_converter.py [-h] [-i INPUT] [-p PARSER] [-a ACTION] [-s SPLIT]
                            [-k AGGREGATE [AGGREGATE ...]] [-v VERBOSE_LEVEL] [-o OUTPUT]
                            [-g GENERATOR]
```
### Arguments
  - `{-h,--help}` -- display help message and exit.
  - `{-i,--input} STRING` -- input file with verbose log (default: `stdin`).
  - `{-p,--parser} oneDNN [default], ...` -- type of parser.
            Refer to ``Parsers`` below.
  - `{-a,--action} generate [default], ...` -- an action.
            Refer to ``Actions`` below.
  - `{-s,--split} BOOL` -- if `true`, generated inputs will be split between
            primitive kinds. Default is `false`.
  - `{-k,--aggregate} field [field ...]` list of fields to use to
            aggregate statistics over for the `breakdown` generator (default: all fields).
            Possible field values: `engine`, `prim_kind`, `impl`, `prop_kind`,
            `mds`, `exts`, `alg_kind`, `shapes`
  - `{-v,--verbose_level} N` -- verbose level. Default is `0`.
            Refer to ``Verbose`` below.
  - `{-o,--output} STRING` -- output file. Default is `stdout`. If output file
            is provided and option `-s` is set, output will be split into
            multiple files with names `driver.output`, where `driver` is a name
            of particular driver.
  - `{-g,--generator} benchdnn [default], ...` target generator.
            Refer to ``Generators`` below.

### Option 2: as Python module
``` python
import verbose_converter

output = verbose_converter.convert(verbose_level, parser, input, action,
         generator, split_output)
```
### Arguments
  - `input STRING` -- string with verbose log.
  - `parser STRING` -- type of parser.
            Refer to ``Parsers`` below.
  - `action STRING` -- an action.
            Refer to ``Actions`` below.
  - `split BOOL` -- if `true`, generated inputs will be split between
            primitive kinds. Default is `false`.
  - `verbose_level N` -- verbose level.
            Refer to ``Verbose`` below.
  - `generator STRING` -- target generator.
            Refer to ``Generators`` below.

### Return value
  - `status STRING` -- status of conversion.
            Refer to ``Statuses`` below.
  - `data` -- if `split` is `true` data is a dictionary where key is name of
            primitive. if `split` is `false` data is a list.

### Actions

| Action    | Description                             |
|:--------- |:-----------                             |
| generate  | generate input using selected generator |
| dumpIR    | dump IR generated after parsing input   |

### Generators

| Generator | Output                            |
|:----------|:----------------------------------|
| benchdnn  | benchdnn test cases               |
| breakdown | breakdown of execution statistics |

#### Benchdnn generator
The benchdnn generator outputs test cases for benchdnn

```
> echo "onednn_verbose,exec,cpu,convolution,jit:avx2,forward_training,src_f32::blocked:acb:f0 wei_f32::blocked:aBdc8b:f0 bia_f32::blocked:a:f0 dst_f32::blocked:acb:f0,,alg:convolution_direct,g1mb2_ic3oc16_iw5ow5kw3sw1dw0pw1" | ./scripts/verbose_converter/verbose_converter.py
--conv --reset --allow-enum-tags-only=0 --engine=cpu --dir=FWD_B --alg=direct --cfg=f32  --stag=acb --wtag=aBdc8b --dtag=acb   g1mb2_ic3oc16_iw5ow5kw3sw1dw0pw1
```

As an example, if we collect the verbose output of the `primitives-convolution-cpp` example
```
> ONEDNN_VERBOSE=1 primitives-convolution-cpp > input.log
```

If we want to validate multiple primitives they should be split by driver type
prior to benchdnn:

```
> ./scripts/verbose_converter/verbose_converter.py -i input.log -s True
--reorder
 --reset --allow-enum-tags-only=0 --engine=cpu    --sdt=f32 --ddt=f32  --stag=abcd --dtag=aBcd8b   3x32x13x13
 --reset --allow-enum-tags-only=0 --engine=cpu    --sdt=f32 --ddt=f32  --stag=abcd --dtag=ABcd8b8a   64x32x3x3
 --reset --allow-enum-tags-only=0 --engine=cpu    --sdt=f32 --ddt=f32  --stag=aBcd8b --dtag=abcd   3x64x4x4
 --reset --allow-enum-tags-only=0 --engine=cpu    --sdt=f32 --ddt=f32  --stag=abcd --dtag=aBcd8b   3x32x13x13
 --reset --allow-enum-tags-only=0 --engine=cpu    --sdt=f32 --ddt=f32  --stag=abcde --dtag=Abcde8a   32x1x1x3x3
 --reset --allow-enum-tags-only=0 --engine=cpu    --sdt=f32 --ddt=f32  --stag=aBcd8b --dtag=abcd   3x32x4x4

--conv
 --reset --allow-enum-tags-only=0 --engine=cpu --dir=FWD_B --alg=direct --cfg=f32  --stag=aBcd8b --wtag=ABcd8b8a --dtag=aBcd8b  --attr-post-ops=eltwise_relu mb3_ic32oc64_ih13oh4kh3sh4dh0ph1_iw13ow4kw3sw4dw0pw1
 --reset --allow-enum-tags-only=0 --engine=cpu --dir=FWD_B --alg=direct --cfg=f32  --stag=aBcd8b --wtag=Abcde8a --dtag=aBcd8b  --attr-post-ops=eltwise_relu g32mb3_ic32oc32_ih13oh4kh3sh4dh0ph1_iw13ow4kw3sw4dw0pw1
```

#### Breakdown generator
The breakdown generator outputs a csv table, where statistics (number
of occurences and timings) are aggregated according to the
`--aggregate` flag option.
As an example, if we collect the verbose output of the `cnn-inference` example
```
> ONEDNN_VERBOSE=1 cnn-inference-f32-cpp > input.log
```

We can gather statistics on `prim_kind` to see which primitive kind is the
most time consuming for this run. This will accumulate the number of calls,
and the timings information for all calls with the same primitive kind.
The output is sorted by highest total time to lowest.
```
> python3  ./scripts/verbose_converter/verbose_converter.py -i input.log -g breakdown -k prim_kind | column -t -s,
prim_kind      ncalls  time(ms)  overall%  agg_ncalls(avg)  agg_time(ms)  agg_overall%
inner_product  300     311.48    45.40     300.00           311.48        45.40
convolution    500     137.46    20.03     400.00           448.94        65.43
eltwise        500     79.14     11.53     433.33           528.08        76.97
reorder        307     71.82     10.47     401.75           599.90        87.44
pooling        300     53.41     7.78      381.40           653.31        95.22
lrn            200     32.79     4.78      351.17           686.10        100.00
```

If we want more details, we can further break that down by shapes as well. So
all primitives with the same primitive kind and shapes, will have they count
call and timings accumulated into one line of the output as follow:
```
> python3  ./scripts/verbose_converter/verbose_converter.py -i input.log -g breakdown -k prim_kind shapes | column -t -s,
prim_kind      shapes                                                      ncalls  time(ms)  overall%  agg_ncalls(avg)  agg_time(ms)  agg_overall%
inner_product  mb1ic256ih6iw6oc4096                                        100     175.46    25.57     100.00           175.46        25.57
inner_product  mb1ic4096oc4096                                             100     89.37     13.03     100.00           264.83        38.60
inner_product  mb1ic4096oc1000                                             100     46.65     6.80      100.00           311.48        45.40
convolution    mb1_ic3oc96_ih227oh55kh11sh4dh0ph0_iw227ow55kw11sw4dw0pw0   100     32.22     4.70      100.00           343.70        50.10
convolution    g2mb1_ic96oc256_ih27oh27kh5sh1dh0ph2_iw27ow27kw5sw1dw0pw2   100     31.47     4.59      100.00           375.17        54.68
eltwise        1x384x13x13                                                 200     31.43     4.58      116.67           406.60        59.26
convolution    mb1_ic256oc384_ih13oh13kh3sh1dh0ph1_iw13ow13kw3sw1dw0pw1    100     26.86     3.91      114.29           433.46        63.18
reorder        96x3x11x11                                                  1       26.72     3.89      100.12           460.18        67.07
convolution    g2mb1_ic384oc384_ih13oh13kh3sh1dh0ph1_iw13ow13kw3sw1dw0pw1  100     24.56     3.58      100.11           484.74        70.65
convolution    g2mb1_ic384oc256_ih13oh13kh3sh1dh0ph1_iw13ow13kw3sw1dw0pw1  100     22.34     3.26      100.10           507.09        73.91
pooling        mb1ic256_ih27oh13kh3sh2dh0ph0_iw27ow13kw3sw2dw0pw0          100     20.23     2.95      100.09           527.32        76.86
pooling        mb1ic96_ih55oh27kh3sh2dh0ph0_iw55ow27kw3sw2dw0pw0           100     18.08     2.64      100.08           545.40        79.49
lrn            mb1ic96ih55iw55ls5beta0.75                                  100     16.90     2.46      100.08           562.30        81.96
reorder        1x3x227x227                                                 100     16.66     2.43      100.07           578.96        84.38
eltwise        1x256x27x27                                                 100     16.48     2.40      100.07           595.44        86.79
eltwise        1x96x55x55                                                  100     16.23     2.37      100.06           611.67        89.15
lrn            mb1ic256ih27iw27ls5beta0.75                                 100     15.89     2.32      100.06           627.56        91.47
reorder        1x256x6x6                                                   100     15.19     2.21      100.06           642.75        93.68
pooling        mb1ic256_ih13oh6kh3sh2dh0ph0_iw13ow6kw3sw2dw0pw0            100     15.10     2.20      100.05           657.85        95.88
eltwise        1x256x13x13                                                 100     15.00     2.19      100.05           672.85        98.07
reorder        4096x4096                                                   1       8.29      1.21      95.33            681.14        99.28
reorder        1000x4096                                                   1       2.11      0.31      91.05            683.25        99.58
reorder        384x256x3x3                                                 1       0.96      0.14      87.13            684.21        99.73
reorder        2x128x192x3x3                                               1       0.72      0.10      83.54            684.93        99.83
reorder        2x192x192x3x3                                               1       0.52      0.08      80.24            685.45        99.91
reorder        2x128x48x5x5                                                1       0.46      0.07      77.19            685.91        99.97
reorder        1x1000                                                      100     0.18      0.03      78.04            686.10        100.00
```


### Parsers

| Parser | Input          |
|:------ |:-----          |
| oneDNN | oneDNN verbose |

### Statuses

| Status  | Value |
|:------  |:----- |
| SUCCESS | 0     |
| FAILED  | 1     |

### Verbose

| Level | Description                         |
|:----- |:-----------                         |
| 0     | no verbose                          |
| 1     | print verbose information to stdout |
