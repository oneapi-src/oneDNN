# Introduction

## C Examples

* `simple_pattern_tiny.c` : a simple and complete pipeline to show the basic usage of oneDNN Graph C API

* `simple_parrten.c` : a simulation of FWK integration, which simulates a simple FWK graph first and then invoke oneDNN Graph C API to optimize and run it

## Cpp Examples

* `simple_parrten.cpp` : a simple and complete pipeline to show the basic usage of oneDNN Graph Cpp API, just like `simple_pattern_tiny.c`

## Usage

* To run the c example

```shell
cd llga
mkdir build && cd build
cmake .. -DDNNL_GRAPH_BUILD_EXAMPLES=1
make -j
./examples/c/simple_pattern_tiny_c
```

* To run the cpp example

```shell
cd llga
mkdir build && cd build
cmake .. -DDNNL_GRAPH_BUILD_EXAMPLES=1
make -j
./examples/cpp/simple_pattern_cpp
```

* To view the oneDNN primitive verbose log

```shell
DNNL_VERBOSE=2 ./examples/c/simple_pattern_tiny_c
```
or
```shell
DNNL_VERBOSE=2 ./examples/cpp/simple_pattern_cpp
```
