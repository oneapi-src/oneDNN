Build from Source {#dev_guide_build}
====================================

## Download the Source Code

Download [DNNL source code](https://github.com/oneapi-src/oneDNN/archive/master.zip)
or clone [the repository](https://github.com/oneapi-src/oneDNN.git).

~~~sh
git clone https://github.com/oneapi-src/oneDNN.git
~~~

## Build the Library

Ensure that all software dependencies are in place and have at least the
minimal supported version.

The DNNL build system is based on CMake. Use

- `CMAKE_INSTALL_PREFIX` to control the library installation location,

- `CMAKE_BUILD_TYPE` to select between build type (`Release`, `Debug`,
  `RelWithDebInfo`).

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
msbuild "DNNL.sln" /p:Configuration=Release /m
  ~~~

- Build the documentation
~~~bat
cmake --build . --target DOC
~~~

- Install the library, headers, and documentation:
~~~bat
cmake --build . --target INSTALL
~~~

## Validate the Build

Run unit tests:

~~~sh
ctest
~~~
