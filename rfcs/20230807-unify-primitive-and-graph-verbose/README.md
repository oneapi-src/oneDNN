# Unify Primitive and Graph API Verbose (RFC)

## Introduction

The goal for this RFC is to demonstrate an unified approach for enabling graph
verbose functionality that can capture information from [Graph
API](https://oneapi-src.github.io/oneDNN/graph_extension.html) and present the
results in oneDNN verbose output.

## Current Graph Verbose

Similar with primitive API, graph API provides an environment variable based
interface `ONEDNN_GRAPH_VERBOSE` shown below for users to profile an application
or troubleshoot API usage errors.

| **Value**       | **Description**                                              |
| --------------- | ------------------------------------------------------------ |
| 0               | No verbose output                                            |
| 1(profile_exec) | Information of compiled partition  execution                 |
| 2(profile)      | Information of partition compilation and  compiled partition execution |
| profile_compile | Information of partition compilation                         |
| check           | Information of some verifying info                           |
| error           | Information of some runtime error                            |
| all             | All information above                                        |

Take `profile` as an example, users can collect the profiling information of
partition compilation and compiled partition execution by specifying
`ONEDNN_GRAPH_VERBOSE=profile`. Verbose log containing marker, operation,
engine_kind, partition_id, partition_kind, op_names, data_formats,
logical_tensors, fpmath_mode, backend_name, time will be printed in the similar
format as oneDNN primitive verbose.

```bash
CMD: ONEDNN_GRAPH_VERBOSE=profile ./tests/benchdnn/benchdnn --graph --case=pattern/f32/conv_bias_add_fusion.json
LOG:
onednn_graph_verbose,info,oneDNN v3.2.0 (commit ce0e4a11be3fa932bdc24f67dceff4df134d0207)
onednn_graph_verbose,info,cpu,runtime:OpenMP,nthr:128
onednn_graph_verbose,info,cpu,isa:Intel AVX-512 with Intel DL Boost
onednn_graph_verbose,info,gpu,runtime:none
onednn_graph_verbose,info,backend,0:dnnl_backend
onednn_graph_verbose,compile:cache_miss,cpu,100006,convolution_post_ops,CONV_0;BINARY_1,data:NCX; filter:OIX;;,in0_f32:0:strided:undef:50x64x56x56:200704s3136s56s1 in1_f32:1:strided:constant:64x64x1x1:64s1s1s1 in2_f32:2:strided:constant:64:1 in3_f32:3:strided:undef:1x1x1x1:1s1s1s1 out0_f32:2209:strided:undef:50x64x56x56:200704s3136s56s1,fpm:strict,dnnl_backend,66.7661
onednn_graph_verbose,exec,cpu,100006,convolution_post_ops,CONV_0;BINARY_1,data:NCX; filter:OIX;;,in0_f32:0:strided:undef:50x64x56x56:200704s3136s56s1 in1_f32:1:strided:constant:64x64x1x1:64s1s1s1 in2_f32:2:strided:constant:64:1 in3_f32:3:strided:undef:1x1x1x1:1s1s1s1 out0_f32:2209:strided:undef:50x64x56x56:200704s3136s56s1,fpm:strict,dnnl_backend,18.824
```

## Motivation

oneDNN Graph API feature has been enabled and built by default in oneDNN's build
system and it is general available since oneDNN v3.1 release. From the user
perspective, a single unified verbose interface can be easier and
straightforward to control the verbose mode for both primitive and graph APIs.
From the implementation perspective, unifying them will save lots of duplicated
codes in the graph API implementation and it will be better to maintain and
align with the changes from primitive verbose.

## Proposal

This proposal is to unify the user interface, verbose format and verbose header
with primitive verbose.

### Unified User Interface

To better unify the user interface with primitive verbose, this proposal is to
leverage `ONEDNN_VERBOSE`:

- `ONEDNN_GRAPH_VERBOSE`  will be deprecated and graph verbose can be controlled
  via `ONEDNN_VERBOSE`
- When `ONEDNN_VERBOSE=1`, verbose of both primitive and compiled partition exec
  profiling will be printed
- When `ONEDNN_VERBOSE=2`, verbose of both primitive creation/execution,
  partition compilation and compiled partition execution profiling will be
  printed
- string based flag `profile`, `profile_exec`, `profile_create`, `check`,
  `error` and `debuginfo` can be used to control verbose for both primitive and
  graph API

Verbose from primitive and graph API will also be printed if user set the
verbose level through `dnnl_set_verbose` API.

To achieve this, `get_verbose()` and `verbose_t` need to be extended and
`get_graph_verbose()` will be removed.

### Verbose Filter

Sometimes, users may only need the graph or primitive verbose. To achieve this,
this proposal introduces verbose filter. User can specify the filter along with
the verbose flag in the form of regexp. e.g.
`ONEDNN_VERBOSE=profile,filter=prim`.

Since oneDNN verbose is designed to be used by frontend users, this proposal
prefers to support filtering the verbose by the components from public API, such
as `primitive`, `graph`, `convolution`, `matmul`, etc. Currently, the
implementation of this RFC have an initial support for filtering profile verbose
by `primitive`, `graph`, `gemm_api` or
[primitive::kind](https://github.com/oneapi-src/oneDNN/blob/master/include/oneapi/dnnl/dnnl.hpp).
e.g.

- `ONEDNN_VERBOSE=profile_exec,filter=graph` will print verbose of
  compiled_partition execution profiling from graph API
- `ONEDNN_VERBOSE=profile_exec,filter=prim` will print verbose of primitive
  execution profiling from primitive API
- `ONEDNN_VERBOSE=profile_exec,filter=conv\|matmul` will print execution
  profiling verbose of (de)convolution and matmul primitive
- If no filters specified, all the verbose will be printed
- If unsupported filters specified, none verbose will be printed

### Unified Verbose Format

To better unify the message log with primitive log, this proposal is to align
with the marker `onednn_verbose` and add a new column `primitive/graph` to
indicate where the verbose came from. Take execution profiling as an example, we
can see the first line came from primitive and the second came from graph API:

```bash
# primitive execution verbose
onednn_verbose,primitive,exec,cpu,convolution,brgconv_1x1:avx512_core,forward_inference,src_f32::blocked:acdb::f0 wei_f32:a:blocked:Acdb64a::f0 bia_f32:a:blocked:a::f0 dst_f32::blocked:acdb::f0,attr-scratchpad:user attr-post-ops:eltwise_relu ,alg:convolution_direct,mb2_ic32oc64_ih112oh112kh1sh1dh0ph0_iw112ow112kw1sw1dw0pw0,1.15503

# compiled partition execution verbose
onednn_verbose,graph,exec,cpu,100006,convolution_post_ops,CONV_0;ELTWISE_1,data:NCX; filter:OIX;;,in0_f32:0:strided:undef:2x32x112x112:401408s12544s112s1 in1_f32:1:strided:constant:64x32x1x1:32s1s1s1 in2_f32:2:strided:constant:64:1 out0_f32:4:strided:undef:2x64x112x112:802816s12544s112s1,fpm:strict,dnnl_backend,5.44092
```

Also, like the primitive API, the graph API also supports some checking info and
the msg log will be shown as:

```bash
# primitive check for invalid u8f32f32 matmul primitive
onednn_verbose,primitive,create:check,matmul,invalid datatype for accumulation,src/common/matmul.cpp:188

# graph check for invalid u8f32f32 matmul op
onednn_verbose,graph,graph:check,add_op,MatMul,given data type for input0 is u8 v.s. expected {f16,bf16,f32},src/graph/interface/op_schema.cpp:250
```

To achieve this, graph related msg need to be added in `verbose_msg.hpp` and
`VPROF/VCHECK/VINFO` need to be adjusted. Duplicated macros like
`VPROFGRAPH/VCHECKGRAPH/VINFOGRAPH` can be removed.

### Unified Verbose Header

Verbose header will be extended for showing infos from graph API. Registered
backends and the graph verbose template can be printed.

```bash
onednn_verbose,info,oneDNN v3.2.0 (commit 01630c9b5795d34d077c228b9e7cd89a72d6b5e5)
onednn_verbose,info,cpu,runtime:OpenMP,nthr:128
onednn_verbose,info,cpu,isa:Intel AVX-512 with Intel DL Boost
onednn_verbose,info,gpu,runtime:none
onednn_verbose,info,graph,backend,0:dnnl_backend
onednn_verbose,primitive,info,template:operation,engine,primitive,implementation,prop_kind,memory_descriptors,attributes,auxiliary,problem_desc,exec_time
onednn_verbose,graph,info,template:operation,engine,partition_id,partition_kind,op_names,data_formats,logical_tensors,fpmath_mode,backend,exec_time
```

## API and implementation

The changes will only impact the implementation and keep both primitive and
graph APIs unchanged.

## Verbose Converter

Currently, verbose from graph API cannot be converted to benchdnn command lines
through verbose converter. With the changes mentioned above, minor changes need
happen in `verbose_converter` side to only convert verbose from primitive API.

## Release

It's proposed to implement the changes on oneDNN master branch and release them
in oneDNN v3.3 as a general available feature.

(EOD)
