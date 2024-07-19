# Proposal for a Logging Mechanism in oneDNN

## Motivation

This is a proposal to introduce a logging mechanism within the oneDNN framework to direct, save and manage the verbose outputs generated from oneDNN into user-specified logs.
Presently, oneDNN in the verbose mode prints all information directly to the console using `stdout` - with a logging mechanism, the user will be able to create custom logs by controlling what and how verbose information is saved in the user-specified logfiles. 

## Objectives 
The logging mechanism will be introduced in oneDNN with the following features:
- deploy logging support as an experimental feature enabled by setting EXPERIMENTAL_LOGGER=ON. 
- add runtime control variables to specify logging options like logfile path and logging levels.
- implement formatting specifications for verbose outputs printed to the log files. 

## Proposal
The proposal is to implement logging support in oneDNN with the help of [`spdlog`]((https://github.com/gabime/spdlog)), a header-only C++ logging library which provides a Python-like formatting API using the bundled [fmt](https://github.com/fmtlib/fmt) lib. The library uses an MIT license, has no cross-component dependencies and has a distinct performance [gain](https://github.com/gabime/spdlog/tree/v1.x?tab=readme-ov-file#benchmarks) over `sprintf` for logging. Key considerations for the proposal are listed as follows:

### 1. Build-time Variables - ONEDNN_EXPERIMENTAL_LOGGING 
`spdlog` can be built with oneDNN using a header-only approach by adding the library headers to the build tree. A pre-compiled version is recommended by the authors to reduce compilation overhead but is not required since the implementation will utilize only limited features from the entire scope of `spdlog`. 

Logging options in oneDNN will be enabled as an experimental feature by setting the variable `ONEDNN_EXPERIMENTAL_LOGGING=ON` during build-time. This will also take an header-based apporach by building the fmtlib library and spdlog headers into the code to enable the logger mechanism.

### 2. Runtime Logging Controls - ONEDNN_VERBOSE_LOGFILE, ONEDNN_VERBOSE_LOGFILE_SIZE, ONEDNN_VERBOSE_NUM_LOGFILES 
A basic requirement for implementing logging support will be to define the run-time variables which the user can specify to manage oneDNN data logging: 
- For the simple case where the logging mechanism involves directly dumping the verbose outputs into a logfile, this can be accomplished with one control variable for specifying the logfile path (`ONEDNN_VERBOSE_LOGFILE=/path/to/file`).
Specifying `ONEDNN_VERBOSE_LOGFILE` automatically enables logging of the verbose output to the user-specified file while in the default case, the data is directly printed to `stdout`.
In this scenario, the data recorded in the logfile mirrors the printed verbose mode information, hence, the logged data can be managed using oneDNN [runtime controls](https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html?highlight=onednn_verbose#run-time-controls) for the verbose mode.
- By default, data will be recorded using a rotating lazy logger with a file size specified by `ONEDNN_VERBOSE_LOGFILE_SIZE(=1024*1024*50)` and the number of rotating files specified by `ONEDNN_VERBOSE_NUM_LOGFILES(=5)`.
- An additional runtime variable `ONEDNN_VERBOSE_LOG_WITH_CONSOLE` will also the user to print to both the logfiles and to `stdout`. 

### 3. Alignment and Specification of Logging Levels
`spdlog` defines the following levels for data logging in its implementation:
```
#define SPDLOG_LEVEL_TRACE 0
#define SPDLOG_LEVEL_DEBUG 1
#define SPDLOG_LEVEL_INFO 2
#define SPDLOG_LEVEL_WARN 3
#define SPDLOG_LEVEL_ERROR 4
#define SPDLOG_LEVEL_CRITICAL 5
#define SPDLOG_LEVEL_OFF 6 
```

The type of tracing information logged for each of these levels is evident from their names. Comparing with the different [verbose modes](https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html?highlight=onednn_verbose#run-time-controls) defined for oneDNN, these logging levels can be aligned with the verbose modes as follows:

|   | `spdlog` Level        | oneDNN Verbose Mode                                              |
|---|-----------------------|------------------------------------------------------------------|
| 0 | SPDLOG_LEVEL_TRACE    | `all`                                                            |
| 1 | SPDLOG_LEVEL_DEBUG    | `debuginfo=<level>`                                              |
| 2 | SPDLOG_LEVEL_INFO     | `dispatch`, `profile_create`, `profile_exec`, `profile`          |
| 3 | SPDLOG_LEVEL_WARNING  | ---                                                              |
| 4 | SPDLOG_LEVEL_ERROR    | `check`                                                          |
| 5 | SPDLOG_LEVEL_CRITICAL | `error`                                                          |
| 6 | SPDLOG_LEVEL_OFF      | `none`                                                           |

With this alignment, the tracing information printed out for each verbose mode can be also logged at the aligned level. Obviously, the logging level here is determined from the value of the `ONEDNN_VERBOSE` variable. 

### 4. Formatting Specifications for Logging
The printed verbose information when `ONEDNN_VERBOSE=all` is formatted and contains the following fields as described [here](https://oneapi-src.github.io/oneDNN/dev_guide_verbose.html#decrypting-the-output):
```
onednn_verbose,primitive,info,template:timestamp,operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
onednn_verbose,graph,info,template:timestamp,operation,engine,partition_id,partition_kind,op_names,data_formats,logical_tensors,fpmath_mode,backend,exec_time
onednn_verbose,1693533460193.346924,primitive,create:cache_miss,cpu,convolution,jit:avx512_core,forward_training,src_f32:a:blocked:aBcd16b::f0 wei_f32:a:blocked:ABcd16b16a::f0 bia_f32:a:blocked:a::f0 dst_f32:a:blocked:aBcd16b::f0,,alg:convolution_direct,mb2_ic16oc16_ih7oh7kh5sh1dh0ph2_iw7ow7kw5sw1dw0pw2,0.709961
```

When logging is enabled, the data will be recorded in the logfiles with the same format and field with the following spdlog field prepended to the verbose output:

```
[%Y-%M-%D %H:%M:%S][onednn][log_level] verbose field 0,verbose field 1, ...
```

#### Example Log:
```
[2024-07-08 17:41:40.417] [oneDNN] [info] -------------------------------------------------
[2024-07-08 17:41:40.417] [oneDNN] [info] logger enabled,logfile:./logs/test_logger.log,size::52428800,amt::5
[2024-07-08 17:41:40.417] [oneDNN] [info] -------------------------------------------------
...
[2024-07-08 17:41:40.427] [oneDNN] [info] onednn_verbose,info,cpu,runtime:OpenMP,nthr:224
[2024-07-08 17:41:40.427] [oneDNN] [info] onednn_verbose,info,cpu,isa:Intel AVX-512 with float16, Intel DL Boost and bfloat16 support 
...
[2024-07-08 17:41:40.427] [oneDNN] [info] onednn_verbose,info,experimental features are enabled
[2024-07-08 17:41:40.427] [oneDNN] [info] onednn_verbose,primitive,info,template:
[2024-07-08 17:41:40.427] [oneDNN] [info] operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
[2024-07-08 17:41:40.427] [oneDNN] [info] onednn_verbose,info,experimental functionality for logging is enabled
[2024-07-08 17:41:40.427] [oneDNN] [info] onednn_verbose,graph,info,template:
[2024-07-08 17:41:40.427] [oneDNN] [info] operation,engine,partition_id,partition_kind,op_names,data_formats,logical_tensors,fpmath_mode,backend,exec_time
[2024-07-08 17:41:40.427] [oneDNN] [info] onednn_verbose,primitive,create:dispatch,matmul,cpu,matmul,brg_matmul:avx10_1_512_amx_fp16,undef,src_f32::blocked:abc::f0 wei_f32::blocked:abc::f0 bia_f32::blocked:abc::f0_mask4 dst_f32::blocked:abc::f0,attr-post-ops:eltwise_relu,,3x128x256:3x256x512,unsupported isa,src/cpu/x64/matmul/brgemm_matmul.cpp:101
[2024-07-08 17:41:40.427] [oneDNN] [info] onednn_verbose,primitive,create:dispatch,matmul,cpu,matmul,brg_matmul:avx10_1_512_amx,undef,src_f32::blocked:abc::f0 wei_f32::blocked:abc::f0 bia_f32::blocked:abc::f0_mask4 dst_f32::blocked:abc::f0,attr-post-ops:eltwise_relu,,3x128x256:3x256x512,unsupported isa,src/cpu/x64/matmul/brgemm_matmul.cpp:101
...
```

## Usage
- For normal verbose mode without logging:
```
$ ONEDNN_VERBOSE=all ./examples/primitives-matmul-cpp
```
- For verbose mode with logging:
```
$ ONEDNN_VERBOSE=all ONEDNN_VERBOSE_LOGFILE=~/test_logger.log ./examples/primitives-matmul-cpp
```
- For verbose mode with logging and printing to `stdout`:
```
$ ONEDNN_VERBOSE=all ONEDNN_VERBOSE_LOGFILE=~/test_logger.log ONEDNN_VERBOSE_LOG_WITH_CONSOLE=1 ./examples/primitives-matmul-cpp
```
- With additional controls for logging:
```
$ ONEDNN_VERBOSE=all ONEDNN_VERBOSE_LOGFILE=~/test_logger.log ONEDNN_VERBOSE_LOGFILE_SIZE=102400 ONEDNN_VERBOSE_NUM_LOGFILES=5 ./examples/primitives-matmul-cpp
```

## References

- **spdlog**: https://github.com/gabime/spdlog
- **fmt** lib: https://github.com/fmtlib/fmt 

#
(EOD)
