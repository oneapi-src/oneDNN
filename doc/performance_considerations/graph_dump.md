# Graph Dump {#dev_guide_graph_dump}

oneDNN Graph API supports graph serialization for visualization and benchmarking.

## Build-Time Controls

The graph dump feature only works when `ONEDNN_BUILD_GRAPH` is ON.

| CMake Option                | Supported values (defaults in bold) | Description
| :---                        | :---                                | :---
| ONEDNN_ENABLE_GRAPH_DUMP    | ON, **OFF**                         | Controls dumping (@ref dev_guide_graph_dump) graph artifacts

## Run-Time Controls

When the feature is enabled at build time, users can use `ONEDNN_GRAPH_DUMP` to
control the serialization level. This option accepts setting flags. These flags
can be combined together to make the library dumping different files. For
example, the below setting will generate files containing library graph and
subgraphs in each partition.

| Variable                  | Flags            | Description
| :---                      | :---             |:---
| ONEDNN_GRAPH_DUMP         | (default)        | No graph or subgraph dumped
|                           | graph            | Library graph
|                           | subgraph         | Library subgraph contained in a partition

```bash
ONEDNN_GRAPH_DUMP=graph,subgraph ./examples/cpu-graph-graph-getting-started-cpp
```

This may produce the following outputs:

```markdown
onednn_graph_verbose,info,serialize graph to a json file graph-100001.json
onednn_graph_verbose,info,serialize graph to a json file graph-100001-partitioning.json
onednn_graph_verbose,info,serialize graph to a json file graph-100002-1313609102600373579.json
onednn_graph_verbose,info,serialize graph to a json file graph-100003-12829238476173481280.json
```
