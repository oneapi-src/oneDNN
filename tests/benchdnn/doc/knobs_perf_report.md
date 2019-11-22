# Performance Report

## Usage
```
    [--perf-template={def [default], csv, CUSTOM_TEMPLATE}]
```

where:
 - def -- default template, has problem name, canonical descriptor dump, ops
            (if applied), minimum and average time and GFLOPS (if applied)
 - csv -- comma-separated values style template. Same as default, but dumps
            all descriptor values with comma delimiter.
 - CUSTOM_TEMPLATE -- a specific set of special words supported by **benchdnn**,
            user-defined output. Refer to the list of all supported special
            words below.


**benchdnn** supports both out-of-the-box and custom performance reports.
A custom template should be passed via the command line and consists of terminal
and nonterminal symbols.
Nonterminal symbols are printed as-is.
Descriptions of terminal symbols are given below.
There is also a notion of modifiers (marked as @) that change the meaning of
terminal symbols; for example, the sign '-' means minimum of (in terms of time).
See the table of modifiers below.

> **Caution:** Threads must be pinned in order to get consistent frequency.

Options supported:

| Syntax        | Primitives                                         | Description
| :--           | :--                                                | :--
| %alg%         | Conv, Eltwise, Lrn, Pool, Reorder, RNN             | Primitive algorithm
| %attr%        | Bnorm, Conv, IP, Reorder                           | Primitive attributes
| %axis%        | Concat, Shuffle, Softmax                           | Primitive axis
| %@bw%         | All with ops                                       | Bytes per second (modifier extended)
| %cfg%         | Conv, IP, Pool, RNN                                | Config, describes data types and filling rules
| %@clocks%     | All                                                | Time in clocks (modifier extended)
| %desc%        | All                                                | Problem descriptor (dimensions and other options included)
| %DESC%        | All                                                | CSV-style problem descriptor (mostly dimensions)
| %dir%         | All, except Concat, RNN, Reorder, Sum              | Primitive prop kind
| %dt%          | Bnorm, Eltwise, Lrn, Shuffle, Softmax              | Data type (precision)
| %sdt%/%ddt%   | Concat, Reorder, Sum                               | Src/Dst data types (precision)
| %engine%      | All                                                | Engine kind
| %flags%       | Bnorm, Lnorm, Reorder                              | Primitive flags
| %@flops%      | All with ops                                       | Ops per second (modifier extended)
| %@freq%       | All                                                | Effective cpu frequency computed as clocks[@] / time[@]
| %group%       | Shuffle                                            | Shuffle group
| %name%        | Bnorm, Conv, IP, Lrn, Pool, RNN                    | Problem name
| %@ops%        | All with ops                                       | Number of ops required (padding is not taken into account)
| %prop%        | RNN                                                | RNN properties
| %tag%         | Bnorm, Eltwise, Lnorm, Lrn, Pool, Shuffle, Softmax | Data format tag (physical memory layout)
| %stat_tag%    | Lnorm                                              | Statistics (meand and variance) format tag (physical memory layout)
| %stag%/%dtag% | Concat, Reorder, Sum                               | Src/Dst format tag (physical memory layout)
| %@time%       | All                                                | Time in ms (modifier extended)

Modifiers supported:

| Name  | Description
| :--   | :--
| Time: |
| -     | min (time) -- default
| 0     | avg (time)
| +     | max (time)
|       |
| Unit: |      (1e0) -- default
| K     | Kilo (1e3)
| M     | Mega (1e6)
| G     | Giga (1e9)

Each primitive has its own descriptor type with options supported. Dimensions
description can be found within each primitive hpp-file.


## Examples

Runs a set of inner products measuring performance with 6 seconds per problem
dumping results with a standard performance template:
``` sh
    ./benchdnn --ip --mode=p --max-ms-per-prb=6000 \
               --batch=inputs/ip/ip_all
```
```
Output template: perf,%engine%,%name%,%desc%,%Gops%,%Gfreq%,%-time%,%-Gflops%,%0time%,%0Gflops%
perf,cpu,"resnet:ip1",mb112oc1000ic2048n"resnet:ip1",0.458752,0,0.521729,879.293,0.576451,795.822
```

Runs a set of inner products measuring performance and dumping results in
CSV-style:
``` sh
    ./benchdnn --ip --mode=p --perf-template=csv \
               --batch=inputs/ip/ip_all
```
```
Output template: perf,%engine%,%name%,%dir%,%cfg%,%attr%,%DESC%,%Gops%,%Gfreq%,%-time%,%-Gflops%,%0time%,%0Gflops%
perf,cpu,"resnet:ip1",FWD_B,f32,,112,1000,2048,1,1,0.458752,0,0.520264,881.768,0.564043,813.328
```

Runs a set of inner products measuring performance and dumping custom template -
reporting descriptor, minimum time, and corresponding gigaFLOPs. Note: ',' is
not a special symbol here; any other delimiter can be used:
``` sh
    ./benchdnn --ip --mode=p --perf-template=%desc%,%-time%,%-Gflops% \
               --batch=inputs/ip/ip_all
```
```
Output template: %desc%,%-time%,%-Gflops%
mb112oc1000ic2048n"resnet:ip1",0.521973,878.881
```
