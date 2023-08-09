# Graph Driver

## Usage

``` sh
    ./benchdnn --graph [benchdnn-knobs] [graph-knobs] [graph-case] ...
```

where *graph-knobs* are:

 - `--mb=INT` -- Override minibatch size specified in the JSON file default
    case. When set to `0`, use batch size as defined by the
    individual test case. The option doesn't take effect for
    operations that don't support the `mb` concept. The default is `0`.

 - `--in-shapes=ID:SHAPE[*TAG+ID:SHAPE*TAG+...]` -- Override a shape and
    stride of a graph input tensor that includes `ID` in a graph with `SHAPE` and
    `TAG` values. `SHAPE` and `TAG` are separated by a `*`. Multiple
    inputs may be specified using the `+` delimiter.

    If both `--mb` and `--in-shapes` are set, `--mb` takes precedence
    over `--in-shapes`.

    The shape of internal tensors and graph output tensors are inferred
    by the graph driver automatically. By default, the option value is empty,
    meaning values are taken from the original graph.

    `TAG` means the memory layout of that tensor, represented by a string starting with `a`.
    The order may differ; a different order means a different memory layout that users may
    provide according to their own needs. Assume, for instance, a tensor with shape `[1,32,4,4]`
    & stride `[512,16,4,1]`, the stride of which can be represented as a Tag `abcd`. If users
    want to modify stride to `[128,1,128,32]`, `TAG` should be provided as `acdb`, and the
    stride values will be calculated within the Benchdnn-graph.

    Below are several use options for `--in-shapes`:
    1. Modify shape only: `--in-shapes=ID:SHAPE[+ID:SHAPE...]`. Users can modify
            rank as well. Modifying shape to 1D tensor with shape `[0]` is also included:
            `--in-shapes=ID:0[+ID:0+...]`.
    2. Modify stride only: `--in-shapes=ID:TAG[+ID:TAG...]`.
    3. Modify shape and stride: `--in-shapes=ID:SHAPE[*TAG+ID:SHAPE*TAG+...]`.
            Users can modify rank as well.
    4. Modify rank to 0; that is, a scalar with shape [], which is represented by `-`
            in cml: `--in-shapes=ID:-[+ID:-+...]`.

    Examples are provided below.

 - `--op-attrs=ID:ATTR_STRING[+ID:ATTR_STRING]` -- Override a series attributes
            value of op with `ID` in the graph with `ATTR_STRING` values.
            `ATTR_STRING` is `ATTR_NAME:ATTR_VALUE[*ATTR_NAME:ATTR_VALUE]`.
            Multiple attributes value changes may be specified using the `*`
            delimeter. Multiple ops modification may be specified using the `+`
            delimeter. By default, the option value is empty, meaning values are taken from original graph.

and *graph-case* is a JSON file which is dumped by a library or created from scratch.
It must be passed to the graph driver as `--case=JSON_FILE`. Refer to the JSON
file example at the end of this document.

The oneDNN Graph serialization feature to dump JSON files at runtime may be enabled
by using the `-DONEDNN_ENABLE_GRAPH_DUMP=ON` build time switch. By default, dump is
disabled. When the build option is on, and the `ONEDNN_GRAPH_DUMP=subgraph` environment
variable is specified, the library generates JSON files with partitions
returned.

## Example

Run the demo `conv_post_ops_fusion` partition
[pattern/f32/conv_post_ops_fusion.json](../inputs/graph/pattern/f32/conv_post_ops_fusion.json)
with default input shapes and op attributes:

```shell
./benchdnn --mode=P --graph --case=./tests/benchdnn/inputs/graph/pattern/f32/conv_post_ops_fusion.json
```

If the JSON file is under `tests/benchdnn/inputs/graph`, we can use the relative
path as shown below,

```shell
./benchdnn --mode=P --graph --case=pattern/f32/conv_post_ops_fusion.json
```

Run the demo pattern with new input shapes by using `--in-shapes`:

```shell
# rewrite input shape only
./benchdnn --mode=C --graph --in-shapes=0:2x64x112x112+1:32x64x2x2 --case=pattern/f32/conv_post_ops_fusion.json
# rewrite stride only
./benchdnn --mode=C --graph --in-shapes=0:dcba+1:dcba --case=pattern/f32/conv_post_ops_fusion.json
# rewrite shape and stride
./benchdnn --mode=C --graph --in-shapes=0:2x64x112x112*dcba+1:32x64x2x2*dcba --case=pattern/f32/conv_post_ops_fusion.json
# rewrite rank
./benchdnn --mode=C --graph --in-shapes=0:2x64x112x112x112+1:32x64x2x2x2 --op-attrs=0:strides:1x1x1*pads_begin:0x0x0*pads_end:0x0x0*dilations:1x1x1 --case=pattern/f32/conv_post_ops_fusion.json
# rewrite rank and stride
./benchdnn --mode=C --graph --in-shapes=0:2x64x112x112x112*edcba+1:32x64x2x2x2*edcba --op-attrs=0:strides:1x1x1*pads_begin:0x0x0*pads_end:0x0x0*dilations:1x1x1 --case=pattern/f32/conv_post_ops_fusion.json
# rewrite rank to 0 rank with shape []
./benchdnn --mode=C --graph --in-shapes=0:- --case=op/f32/add.json
# rewrite to 1D tensor with shape [0]
./benchdnn --mode=C --graph --in-shapes=0:0+1:0 --case=op/f32/add.json
```

Run the demo `conv` op
[op/f32/conv_2d.json](../inputs/graph/op/f32/conv_2d.json) with new strides
attribute by using `--op-attrs`:

```shell
./benchdnn --mode=P --graph --op-attrs=,0:strides:4x4 --case=op/f32/conv_2d.json
```

Run a graph demo batch file [test_graph_ci](../inputs/graph/test_graph_ci):

```shell
./benchdnn --mode=P --graph --batch=test_graph_ci
```

Run same demo batch file on the GPU engine:

```shell
./benchdnn --mode=P --engine=gpu --graph --batch=test_graph_ci
```

Use `-v1` to get more test information, such as graph inputs id and shape,
partition numbers, and so on.

```shell
./benchdnn --mode=P -v1 --graph --mb=1,2,3 --case=op/f32/conv_2d.json
```

Use `--mode=C` or `--mode=c` for correctness testing:

```shell
./benchdnn --mode=C --graph --case=op/f32/conv_2d.json
```

## Demo Cases

Demo JSON files are located in [inputs/graph](../inputs/graph), including
partitions (FP32 MLP partition) and single op (Convolution). Different
data type folders for ops and patterns are available. In general, a JSON file is named as
`workload-pattern_name-additional_info.json`. In this scheme, `workload` stands
for workload name, `pattern_name` stands for the fusion pattern returned by the
library, and `additional_info` differentiates cases based on other settings.
A single op JSON file was named with the op name directly.

## JSON File Example
<details>
    <summary>Conv JSON</summary>

~~~json
{
  "version": "0.5.0",
  "engine_kind": "cpu",
  "fpmath_mode": "strict",
  "graph": [
    {
      "id": 0,
      "name": "Convolution",
      "kind": "Convolution",
      "attrs": {
        "strides": {
          "type": "s64[]",
          "value": [
            2,
            2
          ]
        },
        "pads_begin": {
          "type": "s64[]",
          "value": [
            0,
            0
          ]
        },
        "auto_pad": {
          "type": "string",
          "value": "None"
        },
        "data_format": {
          "type": "string",
          "value": "NCX"
        },
        "pads_end": {
          "type": "s64[]",
          "value": [
            -1,
            -1
          ]
        },
        "groups": {
          "type": "s64",
          "value": 1
        },
        "dilations": {
          "type": "s64[]",
          "value": [
            1,
            1
          ]
        },
        "weights_format": {
          "type": "string",
          "value": "OIX"
        }
      },
      "inputs": [
        {
          "id": 0,
          "dtype": "f32",
          "shape": [
            28,
            512,
            28,
            28
          ],
          "stride": [
            401408,
            1,
            14336,
            512
          ],
          "layout_type": "strided",
          "property_type": "undef"
        },
        {
          "id": 1,
          "dtype": "f32",
          "shape": [
            1024,
            512,
            1,
            1
          ],
          "stride": [
            512,
            1,
            1,
            1
          ],
          "layout_type": "strided",
          "property_type": "constant"
        }
      ],
      "outputs": [
        {
          "id": 2,
          "dtype": "f32",
          "shape": [
            28,
            1024,
            14,
            14
          ],
          "stride": [
            200704,
            1,
            14336,
            1024
          ],
          "layout_type": "strided",
          "property_type": "undef"
        }
      ]
    }
  ]
}
~~~
</details>
