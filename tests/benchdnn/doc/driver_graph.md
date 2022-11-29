## Usage

``` sh
    ./benchdnn --graph [benchdnn-knobs] [graph-knobs] [graph-case] ...
```

where *graph-knobs* are:

 - `--mb=INT` -- override minibatch size specified in the json file default
            case. When set to `0`, use minibatch size as defined by the
            individual test case. The option doesn't take an effect for
            operations that don't support `mb` concept. The default is `0`.
 - `--in-shapes=ID:SHAPE[+ID:SHAPE]` -- override a shape of graph input tensor
            with `ID` in a graph with `SHAPE` values. `SHAPE` as 
            `[DIM0_VALUE0-DIM0_VALUE1-DIM0_VALUE2]xDIM1x...`. The `[ ]` delimiter 
            means tensor `ID` has dynamic shape on `DIM0`, and there 3 real values 
            will be given at execution phase one-by-one. Multiple inputs may be
            specified using `+` delimeter. If both `--mb` and `--in-shapes` were
            set, the `--mb` takes precedence over `--in-shapes`. The shape of
            internal tensor and graph output tensor will be inferred by graph
            driver automatically. By default the option value is empty, meaning
            values are taken from original graph.
 - `--op-attrs=ID:ATTR_STRING[+ID:ATTR_STRING]` -- override a series attributes
            value of op with `ID` in the graph with `ATTR_STRING` values.
            `ATTR_STRING` is `ATTR_NAME:ATTR_VALUE[*ATTR_NAME:ATTR_VALUE]`.
            Multiple attributes value change may be specified using `*`
            delimeter. And multiple ops modification may be specified using `+`
            delimeter. By default the option value is empty, meaning values are
            taken from original graph.

and *graph-case* is a json file which dumped by library or create from scratch.
It will be passed to graph driver with `--case=JSON_FILE`. The canonical json
file as:

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
        "filter_format": {
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

oneDNN Graph serialization feature which dumps json files can be enabled by
using `-DDNNL_GRAPH_ENABLE_DUMP=True` build time switch. It will help to
generate the json files by using applications which integrated the oneDNN graph
API with `DNNL_GRAPH_DUMP=subgraph` environment var at runtime.

## Example

Run the demo `MLP` partition
[pattern/mlp_1x100x200x100.json](../inputs/graph/pattern/mlp_1x100x200x100.json)
with default input shapes and op attributes:

```shell
./benchdnn --mode=P --graph --case=./tests/benchdnn/inputs/graph/pattern/mlp_1x100x200x100.json
```

If the json file is under `tests/benchdnn/inputs/graph`, we can use relative
path like below,

```shell
./benchdnn --mode=P --graph --case=pattern/mlp_1x100x200x100.json
```

 Run the demo MLP with new input shapes by using `--in-shapes`:

```shell
./benchdnn --mode=P --graph --in-shapes=0:32x100+4:200x64,0:32x100 --case=pattern/mlp_1x100x200x100.json
```

Run the demo `conv` op [op/conv_2d.json](../inputs/graph/op/conv_2d.json) with
new strides attribute by using `--op-attrs`:

```shell
./benchdnn --mode=P --graph --op-attrs=,0:strides:4x4  --case=op/conv_2d.json
```

Run a graph demo batch file [test_graph_ci](../inputs/graph/test_graph_ci) and
it can be test with below command,

```shell
./benchdnn --mode=P --graph --batch=test_graph_ci
```

Use `--engine` to specify the case run on CPU or GPU, default will run on CPU
engine.

```shell
./benchdnn --mode=P --engine=gpu --graph --batch=test_graph_ci
```

Use `-v1` to get more test information, for example, graph inputs id and shape,
partition numbers and so on.

```shell
./benchdnn --mode=P -v1 --graph --mb=1,2,3  --case=op/conv_2d.json
```

## Demo cases

There are some demo json files in [inputs/graph](../inputs/graph), including
workload partitions (FP32 & INT8 RN50 patterns), partitions (FP32 MLP partition)
and single op (Convolution). In general, the json file named with
`workload-pattern_name-additional_info.json`. The `workload` means the partition
comes from which workload, the `pattern_name` aligned with the hit fusion
pattern by library, and the `additional_info` used to distinguish different
partitions which are come from same `workload` with same `pattern_name`.
