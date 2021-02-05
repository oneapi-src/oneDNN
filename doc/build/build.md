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
Intel oneAPI Base Toolkit distribution installed to default location you can do
this using `setenv.sh` script
~~~sh
source /opt/intel/oneapi/setvars.sh
~~~

- Configure CMake and generate makefiles
~~~sh
mkdir -p build
cd build

export CC=clang
export CXX=clang++

cmake .. \
          -DDNNL_CPU_RUNTIME=DPCPP
          -DDNNL_GPU_RUNTIME=DPCPP
          <extra build options>
~~~

- Compile with Arm Compute Library (AArch64 only)

~~~sh
export ACL_ROOT_DIR=<path/to/Compute Library>
cmake .. \
         -DDNNL_AARCH64_USE_ACL=ON \
         <extra build options>
~~~
Only ACL versions 20.11 or above are supported. Using ACL versions above	
20.11 may require the `-DCMAKE_CXX_STANDARD=14` and `-DCMAKE_CXX_EXTENSIONS=OFF`
flags to be passed.

#### Build and Install the Library

- Build the library
~~~sh
make -j
~~~

#### GCC targeting AArch64

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

### Windows

#### Microsoft Visual C++ Compiler or Intel C/C++ Compiler

- Generate a Microsoft Visual Studio solution
~~~bat
mkdir build
cd build
cmake -G "Visual Studio 15 2017 Win64" ..
~~~
For the solution to use the Intel C++ Compiler, select the corresponding
toolchain using the cmake `-T` switch:
~~~bat
cmake -G "Visual Studio 15 2017 Win64" -T "Intel C++ Compiler 19.0" ..
~~~

- Build the library
~~~bat
cmake --build .
~~~

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
set CC=clang
set CXX=clang++
cmake .. -G Ninja -DDNNL_CPU_RUNTIME=DPCPP ^
                  -DDNNL_GPU_RUNTIME=DPCPP ^
                  -DCMAKE_PREFIX_PATH=<path to Level Zero headers> ^
                  <extra build options>
~~~
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

## Build documenation

Build the documentation:
~~~sh
cmake --build . --target doc
~~~

## Install library

Install the library, headers, and documentation
~~~sh
cmake --build . --target install
~~~
