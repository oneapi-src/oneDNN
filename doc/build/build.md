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

#### Prepare the Build Space

~~~sh
mkdir -p build && cd build
~~~

#### Generate makefile

- Native compilation:
~~~sh
cmake .. <extra build options>
~~~

- Cross compilation (AArch64 target on Intel 64 host)

~~~sh
export CC=aarch64-linux-gnu-gcc
export CXX=aarch64-linux-gnu-g++
cmake .. \
          -DCMAKE_SYSTEM_NAME=Linux \
          -DCMAKE_SYSTEM_PROCESSOR=AARCH64 \
          -DCMAKE_LIBRARY_PATH=/usr/aarch64-linux-gnu/lib \
          <extra build options>
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

DPCPP runtime requires Intel oneAPI DPC++ Compiler, Intel oneAPI TBB, and
optionally oneAPI Level Zero.

@note If you installed only Intel oneAPI DPC++ Compiler, you can explicitly
specify the path to a compatible version of Intel oneAPI TBB using the
`-DTBBROOT` CMake option.

#### Linux

~~~sh
# Set Intel oneAPI DPC++ Compiler and Intel oneAPI TBB environment.
source /opt/intel/oneapi/setvars.sh
# The command above assumes that the compiler is installed to the default directory.
# If the installation directory was customized, setvars.sh is in the customized directory.

# Set C and C++ compilers
export CC=clang
export CXX=clang++
~~~

##### Build with Support for Level Zero and OpenCL Backends
~~~sh
mkdir build
cd build
cmake -DDNNL_CPU_RUNTIME=DPCPP -DDNNL_GPU_RUNTIME=DPCPP -DCMAKE_PREFIX_PATH=/path/to/level/zero ..
cmake --build .
~~~

##### Build with Support for OpenCL Backend

~~~sh
mkdir build
cd build
cmake -DDNNL_CPU_RUNTIME=DPCPP -DDNNL_GPU_RUNTIME=DPCPP ..
cmake --build .
~~~

#### Windows

~~~bat
:: Set Intel oneAPI DPC++ Compiler and Intel oneAPI TBB environment.
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
:: The command above assumes that the compiler is installed to the default directory.
:: If the installation directory was customized, setvars.bat is in the customized directory.

:: Set C and C++ compilers
set CC=clang
set CXX=clang++
~~~

##### Build with Support for Level Zero and OpenCL Backends

~~~bat
mkdir build
cd build
cmake -G Ninja -DDNNL_CPU_RUNTIME=DPCPP -DDNNL_GPU_RUNTIME=DPCPP -DCMAKE_PREFIX_PATH=/path/to/level/zero ..
cmake --build .
~~~

##### Build with Support for OpenCL Backend
~~~bat
mkdir build
cd build
cmake -G Ninja -DDNNL_CPU_RUNTIME=DPCPP -DDNNL_GPU_RUNTIME=DPCPP ..
cmake --build .
~~~

@note On Windows, oneDNN with DPCPP runtime can be built only with Ninja.
The CC and CXX variables must be set to clang and clang++ respectively and not
to dpcpp.

## Validate the Build

If the library is built for the host system, you can run unit tests using:

~~~sh
ctest
~~~
