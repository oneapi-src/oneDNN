# Graph Serialization {#dev_guide_graph_serialization}

oneDNN Graph supports serializing the graph for visualization and benchmarking.

## Build-Time Controls

| CMake Option                | Supported values (defaults in bold) | Description
| :---                        | :---                                | :---
| DNNL_GRAPH_ENABLE_DUMP      | ON, **OFF**                         | Enables graphs and pattern file dump

## Run-Time Controls

When the feature is enabled at build time, users can use below runtime options
to control the serialization level.

| Variable                  | Value       | Description
| :---                      | :---        |:---
| ONEDNN_GRAPH_DUMP         | 0 (default) | No graph or pattern configuration file dump
|                           | 1           | Library graph and pattern configuration file
|                           | 2           | Library graph, pattern configuration file and serialized subgraph

This option can also accept setting flags. These flags can be combined together
to make the library dumping different files. For example, the below setting will
generate files containing library graph and subgraphs in each partition.

| Variable                  | Flags            | Description
| :---                      | :---             |:---
| ONEDNN_GRAPH_DUMP         | (default)        | No graph or pattern configuration file dump
|                           | graph            | Library graph
|                           | pattern          | Library pattern configuration file
|                           | subgraph         | Library subgraph contained in a partition

```bash
ONEDNN_GRAPH_DUMP=graph,subgraph ./examples/cpp/cpu-get-started-cpp
```

This may produce the following outputs:

```markdown
onednn_graph_verbose,info,serialize graph to a json file graph-100001.json
onednn_graph_verbose,info,serialize graph to a json file graph-100001-partitioning.json
onednn_graph_verbose,info,serialize graph to a json file graph-100002-1313609102600373579.json
onednn_graph_verbose,info,serialize graph to a json file graph-100003-12829238476173481280.json
```
