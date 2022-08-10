# Verbose Mode {#dev_guide_verbose_mode}

oneDNN Graph supports verbose mode which can trace execution of compiled
partition and collection of basic statistic like execution time, input/output
shapes and layouts of compiled partition.

## Build-Time Controls

| CMake Option                | Supported values (defaults in bold) | Description
| :---                        | :---                                | :---
| DNNL_GRAPH_VERBOSE          | **ON**, OFF                         | Enables verbose mode

## Run-Time Controls

When the feature is enabled at build time, users can use below runtime options
to control the verbose level.

| Variable                  | Value       | Description
| :---                      | :---        |:---
| ONEDNN_GRAPH_VERBOSE      | 0 (default) | No verbose output to `stdout`
|                           | 1           | Information of compiled partition execution
|                           | 2           | Information of partition compilation and compiled partition execution

```bash
ONEDNN_GRAPH_VERBOSE=1 ./examples/cpp/cpu-get-started-cpp
```

This may produce the following outputs:

```markdown
onednn_graph_verbose,info,oneDNN Graph v0.5.0 (commit c1fcadbc2bf3914c489fb2c6351e6163d66ad553)
onednn_graph_verbose,info,cpu,runtime:OpenMP,nthr:8
onednn_graph_verbose,info,gpu,runtime:unknown
onednn_graph_verbose,info,backend,0:dnnl_backend
onednn_graph_verbose,exec,cpu,100002,convolution_post_ops,conv0;conv0_bias_add;relu0,data:NCX;NCX; filter:OIX;;;,in0_f32:0:strided:undef:8x3x227x227:154587s51529s227s1
in1_f32:1:strided:undef:96x3x11x11:363s121s11s1 in2_f32:3:strided:undef:96:1 out0_f32:5:strided:undef:8x96x55x55:290400s1s5280s96,fpm:strict,dnnl_backend,3.77686
onednn_graph_verbose,exec,cpu,100003,convolution_post_ops,conv1;conv1_bias_add;relu1,data:NCX;NCX; filter:OIX;;;,
in0_f32:5:strided:undef:8x96x55x55:290400s1s5280s96
in1_f32:6:strided:undef:96x96x1x1:96s1s1s1 in2_f32:8:strided:undef:96:1 out0_f32:10:strided:undef:8x96x55x55:290400s3025s55s1,fpm:strict,dnnl_backend,3.44189
```

The first lines of verbose information, which are denoted with info, contain the
build version and git hash. Each subsequent line of verbose information is
formatted as a comma-separated list contains, in order of appearance in the line
from left to right:

- `onednn_graph_verbose` marker string
- operation: compile:<cache_hit|cache_miss> or exec
- engine kind: cpu or gpu
- partition id: 100002, 100003, etc
- partition kind: convolution_post_ops, matmul_post_ops, etc
- op name: a name list of operators separated with ; which are fused into the
  partition.
- data format: a list of data_format or filter_format separated with `;` of
  operators in the partition.
- information about input/output logical tensors (separated by space):
  - data type prefixed with index
  - layout type: `strided`, `opaque` or `any`
  - constant property: `variable`, `constant` or `undef`
  - shape
  - layout info: stride, layout_id or any
- float-point math mode: `strict`, `bf16`, `f16`, `any` or `tf32`
- backend name
- execution time or compilation time in milliseconds
