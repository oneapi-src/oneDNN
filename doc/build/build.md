# Build from Source {#dev_guide_build}

## Check out the Source Code

Check out the branch from oneDNN repository and update the third party modules.

~~~sh
git clone https://github.com/oneapi-src/oneDNN.git --branch dev-graph-beta-3 --recursive
~~~

## Build the Library

Ensure that all software dependencies are in place and have at least the
minimal supported version.

@note **LLVM 8.0 or later** is required to enable graph compiler.

The oneDNN Graph build system is based on CMake. Use

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

- Compilation with enabled graph compiler

Users can follow the [guidelines
](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)
to build and install LLVM from source, or download and install the pre-built binary
from [here](https://apt.llvm.org/) before generating makefile for graph compiler.

~~~sh
cmake .. -DDNNL_GRAPH_BUILD_COMPILER_BACKEND=True
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

### Windows

- Generate a Microsoft Visual Studio solution

~~~cmd
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019"
~~~

- Build with unittests and examples

~~~cmd
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -DDNNL_GRAPH_BUILD_TESTS=1 -DDNNL_GRAPH_BUILD_EXAMPLES=1 -DCTESTCONFIG_PATH=\\PATH\TO\oneDNNGRAPH\build\src\Release
~~~

- Build with enabled graph compiler

Users can follow the [guidelines
](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)
to build and install LLVM from source, or download and install the pre-built binary
from [here](https://apt.llvm.org/) before generating VS solution for graph compiler.
It's required to link with **Debug** build verion of LLVM in order to use **Debug**
version of graph compiler.

~~~cmd
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -DDNNL_GRAPH_BUILD_COMPILER_BACKEND=1
~~~

- Build the Library

~~~cmd
cmake --build . --config Release
~~~

@note CMake's Microsoft Visual Studio generator does not respect `CMAKE_BUILD_TYPE`
option. Solution file supports both Debug and Release builds with Debug being the
default. You can choose specific build type with `--config` option.

## Validate the Build

If the library is built for the host system, you can run unit tests and examples
using:

~~~sh
cd build
ctest -V
~~~

## Install the Library

To install the built library, you need to have the write privilege of the target
directory with sudo or specifying the target directory via
`-DCMAKE_INSTALL_PREFIX` in the cmake command line.

~~~sh
cd build
make install
~~~
