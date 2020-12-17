# Build from Source {#dev_guide_build}

## Check out the Source Code

Check out the branch from oneDNN repository and update the third party modules.

## Build the Library

Ensure that all software dependencies are in place and have at least the
minimal supported version.

The oneDNN graph build system is based on CMake. Use

- `CMAKE_BUILD_TYPE` to select between build type (`Release`, `Debug`,
  `RelWithDebInfo`).

### Linux/macOS

#### Prepare the Build Space

~~~sh
mkdir -p build && cd build
~~~

#### Generate makefile

- Native compilation:

~~~sh
cmake ..
~~~

- Compilation with enabled unittests and examples

~~~sh
cmake .. -DDNNL_GRAPH_BUILD_EXAMPLES=True -DDNNL_GRAPH_BUILD_TESTS=True
~~~

#### Build and Install the Library

- Build the library:

~~~sh
make -j
~~~

- Build the documentation:

~~~sh
make doc
~~~

## Validate the Build

If the library is built for the host system, you can run unit tests and examples using:

~~~sh
cd build
ctest -V
~~~
