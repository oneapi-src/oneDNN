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

#### GCC, Clang, or Intel C/C++ Compiler

- Set up the environment for the compiler

- Configure CMake and generate makefiles
~~~sh
mkdir -p build
cd build

# Uncomment the following lines to build with Clang
# export CC=clang
# export CXX=clang++

# Uncomment the following lines to build with Intel C/C++ Compiler
# export CC=icc
# export CXX=icpc
cmake .. <extra build options>
~~~

- Build the library
~~~sh
make -j
~~~

#### oneAPI DPC++ Compiler

- Set up the environment for oneAPI DPC++ Compiler. For 
Intel oneAPI Base Toolkit distribution installed to the default location, you can do
this using the `setvars.sh` script:
~~~sh
source /opt/intel/oneapi/setvars.sh
~~~

- Configure CMake and generate makefiles
~~~sh
mkdir -p build
cd build

export CC=icx
export CXX=icpx

cmake .. \
          -DDNNL_CPU_RUNTIME=DPCPP
          -DDNNL_GPU_RUNTIME=DPCPP
          <extra build options>
~~~

Open-source version of oneAPI DPC++ Compiler may not contain OpenCL runtime.
In this case, you can use `OPENCLROOT` CMake option or environment variable
of the same name to specify path to the OpenCL runtime if it is installed in
a custom location.

@note Open-source version of oneAPI DPC++ Compiler does not have the icx/icpx
drivers, use clang/clang++ instead.

- Build the library
~~~sh
make -j
~~~

#### GCC targeting AArch64 on x64 host

- Set up the environment for the compiler

- Configure CMake and generate makefiles
~~~sh
export CC=aarch64-linux-gnu-gcc
export CXX=aarch64-linux-gnu-g++
cmake .. \
          -DCMAKE_SYSTEM_NAME=Linux \
          -DCMAKE_SYSTEM_PROCESSOR=AARCH64 \
          -DCMAKE_LIBRARY_PATH=/usr/aarch64-linux-gnu/lib \
          <extra build options>
~~~

- Build the library
~~~sh
make -j
~~~

#### GCC with Arm Compute Library (ACL) on AArch64 host

- Set up the environment for the compiler

- Configure CMake and generate makefiles
~~~sh
export ACL_ROOT_DIR=<path/to/Compute Library>
/cmake .. \
          -DDNNL_AARCH64_USE_ACL=ON \
          <extra build options>
~~~

- Build the library
~~~sh
make -j
~~~

### Windows

#### Microsoft Visual C++ Compiler or Intel C/C++ Compiler

- Generate a Microsoft Visual Studio solution
~~~bat
mkdir build
cd build
cmake -G "Visual Studio 16 2019 Win64" ..
~~~
For the solution to use the Intel C++ Compiler, select the corresponding
toolchain using the cmake `-T` switch:
~~~bat
cmake -G "Visual Studio 16 2019 Win64" -T "Intel C++ Compiler 19.0" ..
~~~

- Build the library
~~~bat
cmake --build . --config=Release
~~~

@note CMake's Microsoft Visual Studio generator does not respect `CMAKE_BUILD_TYPE` option.
Solution file supports both Debug and Release builds with Debug being the default.
You can choose specific build type with `--config` option.

@note You can also open `oneDNN.sln` to build the project from the
Microsoft Visual Studio IDE.

#### oneAPI DPC++ Compiler

- Set up the environment for oneAPI DPC++ Compiler. For
Intel oneAPI Base Toolkit distribution installed to default location you can do
this using `setvars.bat` script
~~~bat
"C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
~~~
or open `Intel oneAPI Commmand Prompt` instead.

- Download [oneAPI Level Zero headers](https://github.com/oneapi-src/level-zero/releases/tag/v1.0)
from Github and unpack the archive.

- Generate `Ninja` project
~~~bat
mkdir build
cd build

:: Set C and C++ compilers
set CC=icx
set CXX=icx
cmake .. -G Ninja -DDNNL_CPU_RUNTIME=DPCPP ^
                  -DDNNL_GPU_RUNTIME=DPCPP ^
                  -DCMAKE_PREFIX_PATH=<path to Level Zero headers> ^
                  <extra build options>
~~~

Open-source version of oneAPI DPC++ Compiler may not contain OpenCL runtime.
In this case, you can use `OPENCLROOT` CMake option or environment variable
of the same name to specify path to the OpenCL runtime if it is installed in
a custom location.

@note Open-source version of oneAPI DPC++ Compiler does not have the icx driver,
use clang/clang++ instead.

@note The only CMake generator that supports oneAPI DPC++ Compiler on Windows
is Ninja. CC and CXX variables must be set to clang and clang++ respectively. 

- Build the library
~~~bat
cmake --build .
~~~

## Validate the Build

If the library is built for the host system, you can run unit tests using:
~~~sh
ctest
~~~

## Build documentation

- Install the requirements
~~~sh
conda env create -f ../doc/environment.yml
conda activate onednn-doc
~~~

- Build the documentation
~~~sh
cmake --build . --target doc
~~~

## Install library

Install the library, headers, and documentation
~~~sh
cmake --build . --target install
~~~
The install directory is specified by the [CMAKE_INSTALL_PREFIX](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html)
cmake variable. When installing in the default directory, the above command
needs to be run with administrative privileges using `sudo` on Linux/Mac or a
command prompt run as administrator on Windows. 
