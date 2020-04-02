Build from Source {#dev_guide_build}
====================================

## Download the Source Code

Download [oneDNN source code](https://github.com/oneapi-src/oneDNN/archive/master.zip)
or clone [the repository](https://github.com/oneapi-src/oneDNN.git).

~~~sh
git clone https://github.com/oneapi-src/oneDNN.git
~~~

## Build the Library

Ensure that all software dependencies are in place and have at least the
minimal supported version.

The oneDNN build system is based on CMake. Use

- `CMAKE_INSTALL_PREFIX` to control the library installation location,

- `CMAKE_BUILD_TYPE` to select between build type (`Release`, `Debug`,
  `RelWithDebInfo`).

- `CMAKE_PREFIX_PATH` to specify directories to be searched for the
  dependencies located at non-standard locations.

See @ref dev_guide_build_options for detailed description of build-time
configuration options.

### Linux/macOS

- Generate makefile:
~~~sh
mkdir -p build && cd build && cmake ..
~~~

- Build the library:
~~~sh
make -j
~~~

- Build the documentation:
~~~sh
make doc
~~~

- Install the library, headers, and documentation:
~~~sh
make install
~~~

### Windows

- Generate a Microsoft Visual Studio solution:
~~~bat
mkdir build && cd build && cmake -G "Visual Studio 15 2017 Win64" ..
~~~
For the solution to use the Intel C++ Compiler, select the corresponding
toolchain using the cmake `-T` switch:
~~~bat
cmake -G "Visual Studio 15 2017 Win64" -T "Intel C++ Compiler 19.0" ..
~~~

- Build the library:
~~~bat
cmake --build .
~~~
You can also use the `msbuild` command-line tool directly (here
`/p:Configuration` selects the build configuration which can be different from
the one specified in `CMAKE_BUILD_TYPE`, and `/m` enables a parallel build):
~~~bat
msbuild "oneDNN.sln" /p:Configuration=Release /m
  ~~~

- Build the documentation
~~~bat
cmake --build . --target DOC
~~~

- Install the library, headers, and documentation:
~~~bat
cmake --build . --target INSTALL
~~~

### Building with DPCPP runtime

DPCPP runtime requires Intel oneAPI DPC++ Compiler. You can explicitly specify
the path to Intel oneAPI DPC++ Compiler installation using 
`-DDPCPPROOT` CMake option.

C and C++ compilers need to be set to point to Intel oneAPI DPC++ Compilers.

#### Linux

~~~sh
# Set Intel oneAPI DPC++ Compiler environment
# <..>/setvars.sh

# Set C and C++ compilers
export CC=clang
export CXX=clang++

mkdir build
cd build
cmake -DDNNL_CPU_RUNTIME=DPCPP -DDNNL_GPU_RUNTIME=DPCPP ..
cmake --build .
~~~

#### Windows

@note
    Currently, building on Windows has a few limitations:
    - Only the Clang compiler with GNU-like command-line is supported
      (`clang.exe` and `clang++.exe`).
    - Only the Ninja generator is supported.
    - CMake version must be 3.15 or newer.

~~~bat
:: Set Intel oneAPI DPC++ Compiler environment
:: <..>\setvars.bat

:: Set C and C++ compilers (must have GNU-like command-line interface)
set CC=clang
set CXX=clang++

mkdir build
cd build
cmake -G Ninja -DDNNL_CPU_RUNTIME=DPCPP -DDNNL_GPU_RUNTIME=DPCPP ..
cmake --build .
~~~

## Validate the Build

Run unit tests:

~~~sh
ctest
~~~
