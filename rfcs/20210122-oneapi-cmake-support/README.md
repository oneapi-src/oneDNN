# Proposal on CMake Support in oneDNN for oneAPI

## Motivation
Currently, oneDNN provides CMake support to the users meaning that the users
can use `find_package(dnnl)` to integrate oneDNN in their CMake-based build system.

However, the way the support is currently implemented doesn't fulfill the requirements
for CMake support in oneAPI products. As part of the oneAPI releases oneDNN is
distributed in binary form and comes in multiple configurations to support the
customers who want to use oneDNN for various runtimes.

Currently, oneDNN provides the following list of configurations for oneAPI releases:
| Configuration         | Dependency
| :---------------------| :---------
| `cpu_iomp`            | Intel OpenMP runtime
| `cpu_gomp`            | GNU\* OpenMP runtime
| `cpu_vcomp`           | Microsoft Visual C OpenMP runtime
| `cpu_tbb`             | Threading Building Blocks (TBB)
| `cpu_dpcpp_gpu_dpcpp` | [Intel oneAPI DPC++ Compiler](https://software.intel.com/en-us/oneapi/dpc-compiler), TBB, OpenCL runtime, oneAPI Level Zero runtime

According to the current design of CMake support, each of the configurations has
its own CMake config file that can be used for locating oneDNN with `find_package`.
This goes against the design of CMake support for oneAPI products. The main difference
is that there should be only one CMake config file per oneAPI product. Meaning
that oneDNN has to provide a common CMake config file that would work for all
aforementioned configurations.

The rationale behind such design is an ability to use `find_package` without configuring
the environment with oneAPI provided `setvars.sh` and `setvars.bat` scripts.
This is where the requirement about having only one CMake config file is coming
from.

Given that there is no way to automatically specify what oneDNN configuration
to pick when using `find_package` without using those environment scripts, users
have to explicitly specify what oneDNN configuration they need.

There are two options to do that.

The common part of those options is the following:

* The existing CMake support in oneDNN stays intact
* A new, common CMake config file is introduced. It will be used only for oneAPI releases

The differences are described below.

### Option #1 (recommended)

* User specifies a particular oneDNN configuration with a CMake variable `DNNL_CONFIGURATION`
* If the variable is not set the default configuration (currently cpu_dpcpp_gpu_dpcpp) is picked

Pros:
* Coexistence of different CMake configs. Users can always rely on 
`find_package(dnnl)`. In case of non-oneAPI release the `DNNL_CONFIGURATION`
variable will be ignored and the configuration that was built by the user will be used
* Transparent mechanism for choosing oneDNN configuration

Cons:
* A little peculiar. The peculiarity needs to be documented

```cmake
set(DNNL_CONFIGURATION cpu_gomp)
find_package(dnnl)
target_link_libraries(foo DNNL::dnnl)  # DNNL::dnnl provides cpu_gomp configuration


set(DNNL_CONFIGURATION cpu_tbb)
find_package(dnnl)
target_link_libraries(foo DNNL::dnnl)  # DNNL::dnnl provides cpu_tbb configuration
```

### Option #2

* User specifies a particular oneDNN configuration with `COMPONENTS` when using
`find_package`
* If no components are provided the default configuration (currently cpu_dpcpp_gpu_dpcpp) is picked

Pros:
* Using a dedicated CMake mechanism. No oneDNN specific variables

Cons:
* Users cannot always rely on `find_package(dnnl)` and hence cannot switch between
oneAPI and non-oneAPI releases
* Semantics. Components are something that a project consists of, but in case of
oneDNN each component IS the project
* If it will be decided to split oneDNN into different components (libraries)
in the future it will confuse users. Because they will have to use `COMPONENTS` to
specify oneDNN configuration and oneDNN components. For example, TBB provides
several components: TBB::tbb, TBB::tbbmalloc, etc.

```cmake
find_package(dnnl COMPONENTS cpu_gomp)
target_link_libraries(foo DNNL::cpu_gomp)  # DNNL::cpu_gomp provides cpu_gomp configuration

find_package(dnnl COMPONENTS cpu_tbb)
target_link_libraries(foo DNNL::cpu_tbb)  # DNNL::cpu_tbb provides cpu_tbb configuration
```
