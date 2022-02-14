# Environmental Variables {#dev_guide_environmental_variables}

oneDNN Graph supports the following runtime environmental variables. Typically,
users would not need to set or change the values of them.

The support for each environmental variable may depend on cmake build options.
Please refer to [build options](@ref dev_guide_build_options).

| Variable                  | Value       | Description
| :---                      | :---        |:---
| ONEDNN_GRAPH_VERBOSE      | 0 (default) | No verbose output to `stdout`
|                           | 1           | Information of compiled partition execution
|                           | 2           | Information of partition compilation and compiled partition execution
| ONEDNN_GRAPH_DUMP         | 0 (default) | No graph or pattern configuration file dump
|                           | 1           | Library graph and pattern configuration file
|                           | 2           | Library graph, pattern configuration file, and backend subgraph
