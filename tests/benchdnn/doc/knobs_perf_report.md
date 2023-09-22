# Performance Report

## Usage
```
    [--perf-template={def [default], csv, CUSTOM_TEMPLATE}]
```

where:
 - `def` -- default template. Has problem name, problem dump, ops (if applied),
          minimum and average time and GFLOPS (if applied)
 - `csv` -- comma-separated values style template. Same as default, but dumps
          problem and descriptor values with comma delimiter.
 - `CUSTOM_TEMPLATE` -- user-defined template. Should consist of special options
                      supported by specific driver. Refer to the list of
                      options supported below.


**benchdnn** supports both out-of-the-box and custom performance reports.
A custom template should be passed via the command line and consists of terminal
and nonterminal symbols. Nonterminal symbols are printed as-is. Descriptions of
terminal symbols are given below.

> **Note:** Following generalized types are used below:
>
> * 'Data md based' = {Bnorm, Eltwise, Lnorm, Lrn, Prelu, Shuffle, Softmax}
> * 'Problem desc based' = {Bnorm, Conv, IP, Lrn, Matmul, Pool, Resampling, RNN}
> * 'Ops based' = {Conv, IP, Matmul, RNN}

Data types options supported:

| Syntax | Primitives                                     | Description
| :--    | :--                                            | :--
| %cfg%  | Conv, IP, Matmul, Pool, RNN                    | Config which describes data types and filling rules
| %dt%   | Data md based, Resampling, Zeropad             | Source and Destination Data type (precision)
| %ddt%  | Binary, Concat, Reduction, Reorder, Sum        | Destination data types (precision)
| %sdt%  | Binary, Concat, Prelu, Reduction, Reorder, Sum | Source data types (precision)

Format tags (physical memory layout) options supported:

| Syntax     | Primitives                                                       | Description
| :--        | :--                                                              | :--
| %tag%      | Data md based, Pool, Resampling, Zeropad                         | Source and Destination format tag
| %dtag%     | Binary, Concat, Conv, IP, Matmul, Reduction, Reorder, Sum        | Destination format tag
| %stag%     | Binary, Concat, Conv, IP, Matmul, Prelu, Reduction, Reorder, Sum | Source format tag
| %wtag%     | Conv, IP, Matmul                                                 | Weights format tag
| %stat_tag% | Lnorm                                                            | Layer Normalization statistics (mean and variance) format tag

Other problem specific options supported:

| Syntax       | Primitives                                                            | Description
| :--          | :--                                                                   | :--
| %activation% | RNN                                                                   | RNN activation function
| %alg%        | Binary, Conv, Eltwise, Lrn, Pool, Reduction, Reorder, Resampling, RNN | Primitive algorithm
| %attr%       | All                                                                   | Primitive attributes
| %axis%       | Concat, Shuffle, Softmax                                              | Primitive axis
| %desc%       | All                                                                   | String style problem descriptor
| %DESC%       | All                                                                   | CSV-style problem descriptor values only
| %dir%        | All, except Concat, RNN, Reorder, Sum                                 | Primitive prop kind
| %direction%  | RNN                                                                   | RNN direction execution
| %driver%     | All                                                                   | Name of the current driver (e.g. conv, reorder)
| %engine%     | All                                                                   | Engine kind
| %flags%      | Bnorm, Lnorm, Reorder                                                 | Primitive flags
| %group%      | Shuffle                                                               | Shuffle group
| %impl%       | All                                                                   | Library implementation name for a given problem
| %idx%        | All                                                                   | Test index
| %mb%         | Problem desc based, Eltwise, Softmax                                  | Mini-batch value from user input. Prints `0` in case of input `--mb=0`
| %name%       | Problem desc based                                                    | Problem name
| %prb%        | All                                                                   | Canonical problem (options and descriptor in REPRO style)
| %prop%       | RNN                                                                   | RNN prop kind

Performance profiling. All options are modifier extended (see below). Modifiers
change the meaning of terminal symbols. I.e., the sign '-' means minimum of
(in terms of time). Extensions should be specified after first percent,
describing the option in a specific order: first is time modifier, second is
unit modifier. I.e. `%-Gflops%`, not `%G-flops%`.

> **Caution:** Threads must be pinned in order to get consistent frequency.

Performance profiling options supported:

| Syntax     | Primitives | Description
| :--        | :--        | :--
| %@time%    | All        | Execution time in milliseconds
| %@clocks%  | All        | Execution time in clocks
| %@freq%    | All        | Effective CPU frequency computed as `clocks / time`
| %@ibytes%  | All        | Number of input memories bytes of a problem
| %@obytes%  | All        | Number of output memories bytes of a problem
| %@iobytes% | All        | Number of input and output memories bytes of a problem
| %@bw%      | All        | Bandwidth computed as `iobytes / time`
| %@ops%     | Ops based  | Number of ops required (padding is not taken into account)
| %@flops%   | Ops based  | FLOPS computed as `ops / time`
| %@cpdtime% | All        | Primitive descriptor creation time in milliseconds. See `Create Time Notes`.
| %@cptime%  | All        | Primitive creation time in milliseconds. See `Create Time Notes`.
| %@ctime%   | All        | Total creation time (primitive descriptor + primitive) in milliseconds. See `Create Time Notes`.

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

### Create Time Notes

Benchdnn runs two create calls when primitive cache feature is enabled. A timer,
responsible for collecting create milliseconds, catches both cases. A case when
primitive cache was not hit can be obtained through the empty or `max` modifier
(the default). A case when primitive cache was hit can be obtained through the
`min` modifier. The average modifier for create times is not recommended since
this time doesn't represent any specific scenario.

## Examples

Runs a set of inner products measuring performance with 6 seconds per problem
dumping results with a standard performance template:
``` sh
    ./benchdnn --ip --mode=p --max-ms-per-prb=6000 \
               --batch=inputs/ip/test_ip_all
```
```
Output template: perf,%engine%,%name%,%prb%,%Gops%,%Gfreq%,%-time%,%-Gflops%,%0time%,%0Gflops%
perf,cpu,"resnet:ip1",mb112oc1000ic2048n"resnet:ip1",0.458752,0,0.521729,879.293,0.576451,795.822
```

Runs a set of inner products measuring performance and dumping results in
CSV-style:
``` sh
    ./benchdnn --ip --mode=p --perf-template=csv \
               --batch=inputs/ip/test_ip_all
```
```
Output template: perf,%engine%,%name%,%dir%,%cfg%,%attr%,%DESC%,%Gops%,%Gfreq%,%-time%,%-Gflops%,%0time%,%0Gflops%
perf,cpu,"resnet:ip1",FWD_B,f32,,112,1000,2048,1,1,0.458752,0,0.520264,881.768,0.564043,813.328
```

Runs a set of inner products measuring performance and dumping custom template -
reporting descriptor, minimum time, and corresponding gigaFLOPS. Note: ',' is
not a special symbol here; any other delimiter can be used:
``` sh
    ./benchdnn --ip --mode=p --perf-template=%prb%,%-time%,%-Gflops% \
               --batch=inputs/ip/test_ip_all
```
```
Output template: %prb%,%-time%,%-Gflops%
mb112oc1000ic2048n"resnet:ip1",0.521973,878.881
```
