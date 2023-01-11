# Verbose log converter

Verbose log converter is a tool that allows to convert [oneDNN
verbose](https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html)
output to various outputs (input files for benchdnn and execution
statistics breakdown at this time). The tool can be extended to
produce other types of output by adding generators.

## Requirements
 - Python 3.7

## Compatibility
The script is compatible with particular oneDNN version it is distributed with.
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
> python3 verbose_converter.py -i input.log -g breakdown -k prim_kind
prim_kind,ncalls,agg_time(ms),overall%
inner_product,300,416.82,71.71
convolution,500,126.54,21.77
lrn,200,27.90,4.80
pooling,300,3.73,0.64
eltwise,500,3.44,0.59
reorder,205,2.81,0.48
```

If we want more details, we can further break that down by shapes as well. So
all primitives with the same primitive kind and shapes, will have they count
call and timings accumulated into one line of the output as follow:
```
> python3 verbose_converter.py -i input.log -g breakdown -k prim_kind shapes
prim_kind,shapes,ncalls,agg_time(ms),overall%
inner_product,mb1ic256ih6iw6oc4096,100,272.12,46.82
inner_product,mb1ic4096oc4096,100,115.82,19.93
convolution,g2mb1_ic96oc256_ih27oh27kh5sh1dh0ph2_iw27ow27kw5sw1dw0pw2,100,37.31,6.42
convolution,mb1_ic256oc384_ih13oh13kh3sh1dh0ph1_iw13ow13kw3sw1dw0pw1,100,30.69,5.28
inner_product,mb1ic4096oc1000,100,28.89,4.97
convolution,g2mb1_ic384oc384_ih13oh13kh3sh1dh0ph1_iw13ow13kw3sw1dw0pw1,100,23.41,4.03
convolution,mb1_ic3oc96_ih227oh55kh11sh4dh0ph0_iw227ow55kw11sw4dw0pw0,100,19.80,3.41
lrn,mb1ic96ih55iw55ls5beta0.75,100,18.19,3.13
convolution,g2mb1_ic384oc256_ih13oh13kh3sh1dh0ph1_iw13ow13kw3sw1dw0pw1,100,15.32,2.64
lrn,mb1ic256ih27iw27ls5beta0.75,100,9.70,1.67
pooling,mb1ic96_ih55oh27kh3sh2dh0ph0_iw55ow27kw3sw2dw0pw0,100,1.89,0.32
pooling,mb1ic256_ih27oh13kh3sh2dh0ph0_iw27ow13kw3sw2dw0pw0,100,1.31,0.23
eltwise,1x96x55x55,100,1.29,0.22
reorder,96x3x11x11,1,1.18,0.20
eltwise,1x384x13x13,200,0.92,0.16
eltwise,1x256x27x27,100,0.79,0.14
pooling,mb1ic256_ih13oh6kh3sh2dh0ph0_iw13ow6kw3sw2dw0pw0,100,0.53,0.09
reorder,1x256x6x6,100,0.51,0.09
eltwise,1x256x13x13,100,0.44,0.08
reorder,384x256x3x3,1,0.26,0.04
reorder,2x192x192x3x3,1,0.26,0.04
reorder,1x1000,100,0.23,0.04
reorder,2x128x192x3x3,1,0.20,0.04
reorder,2x128x48x5x5,1,0.17,0.03
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
