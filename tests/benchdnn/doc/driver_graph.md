# Graph Driver

## Usage

``` sh
    ./benchdnn --graph [benchdnn-knobs] [graph-knobs] [graph-case] ...
```

where *graph-knobs* are:

 - `--mb=INT` -- Override minibatch size specified in the JSON file default
            case. When set to `0`, use minibatch size as defined by the
            individual test case. The option doesn't take an effect for
            operations that don't support `mb` concept. The default is `0`.
 - `--in-shapes=ID:SHAPE[+ID:SHAPE]` -- Override a shape of graph input tensor
            with `ID` in a graph with `SHAPE` values. Multiple inputs may be
            specified using `+` delimeter. If both `--mb` and `--in-shapes` were
            set, the `--mb` takes precedence over `--in-shapes`. The shape of
            internal tensor and graph output tensor will be inferred by graph
            driver automatically. By default the option value is empty, meaning
            values are taken from original graph.
 - `--op-attrs=ID:ATTR_STRING[+ID:ATTR_STRING]` -- Override a series attributes
            value of op with `ID` in the graph with `ATTR_STRING` values.
            `ATTR_STRING` is `ATTR_NAME:ATTR_VALUE[*ATTR_NAME:ATTR_VALUE]`.
            Multiple attributes value change may be specified using `*`
            delimeter. And multiple ops modification may be specified using `+`
            delimeter. By default the option value is empty, meaning values are
            taken from original graph.

and *graph-case* is a JSON file which dumped by library or created from scratch.
It must be passed to the graph driver as `--case=JSON_FILE`. Refer to a JSON
file example at the end of this document.

oneDNN Graph serialization feature to dump JSON files in runtime may be enabled
by using `-DONEDNN_ENABLE_GRAPH_DUMP=ON` build time switch. By default dump is
disabled. When build option is on, and `ONEDNN_GRAPH_DUMP=subgraph` environment
variable is specified, the library generates JSON files with partitions
returned.

## Example

Run the demo `conv_post_ops_fusion` partition
[pattern/f32/conv_post_ops_fusion.json](../inputs/graph/pattern/f32/conv_post_ops_fusion.json)
with default input shapes and op attributes:

```shell
./benchdnn --mode=P --graph --case=./tests/benchdnn/inputs/graph/pattern/f32/conv_post_ops_fusion.json
```

If the JSON file is under `tests/benchdnn/inputs/graph`, we can use relative
path like below,

```shell
./benchdnn --mode=P --graph --case=pattern/f32/conv_post_ops_fusion.json
```

Run the demo pattern with new input shapes by using `--in-shapes`:

```shell
./benchdnn --mode=P --graph --in-shapes=0:2x64x112x112+1:32x64x2x2 --case=pattern/f32/conv_post_ops_fusion.json
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

Run same demo batch file on GPU engine:

```shell
./benchdnn --mode=P --engine=gpu --graph --batch=test_graph_ci
```

Use `-v1` to get more test information, for example, graph inputs id and shape,
partition numbers and so on.

```shell
./benchdnn --mode=P -v1 --graph --mb=1,2,3 --case=op/f32/conv_2d.json
```

Use `--mode=C` or `--mode=c` for correctness testing:

```shell
./benchdnn --mode=C --graph --case=op/f32/conv_2d.json
```

## Demo Cases

There are some demo JSON files in [inputs/graph](../inputs/graph), including
partitions (FP32 MLP partition) and single op (Convolution). There are different
data type folders for ops and patterns. In general, a JSON file named as
`workload-pattern_name-additional_info.json`. In this scheme `workload` stands
for workload name, `pattern_name` stands for fusion pattern returned by the
library, and `additional_info` differentiates cases based on other settings.
Single op JSON file was named with op name directly.

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
