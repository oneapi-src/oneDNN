Linking to the Library {#dev_guide_link}
===========================================

Deep Neural Network Library (DNNL)
includes several header files providing C and C++ APIs for the functionality
and one or several libraries depending on how DNNL was built.

## Header Files

| File                     | Description
| :---                     | :---
| include/dnnl.h         | C header
| include/dnnl.hpp       | C++ header
| include/dnnl_types.h   | Auxiliary C header
| include/dnnl_config.h  | Auxiliary C header
| include/dnnl_version.h | C header with version information

## Libraries

### Linux

| File                  | Description
| :---                  | :---
| lib/libdnnl.so      | DNNL dynamic library
| lib/libdnnl.a       | DNNL static library (if built with `DNNL_LIBRARY_TYPE=STATIC`)

### macOS

| File                     | Description
| :---                     | :---
| lib/libdnnl.dylib      | DNNL dynamic library
| lib/libdnnl.a          | DNNL static library (if built with `DNNL_LIBRARY_TYPE=STATIC`)

### Windows

| File              | Description
| :---              | :---
| bin\libdnnl.dll | DNNL dynamic library
| lib\libdnnl.lib | DNNL import library

## Linking to DNNL

The examples below assume that DNNL is installed in the directory
defined in the `DNNLROOT` environment variable.

### Linux/macOS

~~~sh
g++ -std=c++11 -I${DNNLROOT}/include -L${DNNLROOT}/lib simple_net.cpp -ldnnl
clang++ -std=c++11 -I${DNNLROOT}/include -L${DNNLROOT}/lib simple_net.cpp -ldnnl
icpc -std=c++11 -I${DNNLROOT}/include -L${DNNLROOT}/lib simple_net.cpp -ldnnl
~~~

@note
Applications linked dynamically will resolve the dependencies at runtime. Make
sure that the dependencies are available in the standard locations defined by
the operating system, in the locations listed in the `LD_LIBRARY_PATH` (Linux)
or `DYLD_LIBRARY_PATH` (macOS) environment variable or the `rpath` mechanism.

### Windows

To link the application from the command line, set up the `LIB` and `INCLUDE`
environment variables to point to the locations of the DNNL headers and
libraries.

~~~bat
icl /I%DNNLROOT%\include /Qstd=c++11 /qopenmp simple_net.cpp %DNNLROOT%\lib\dnnl.lib
cl /I%DNNLROOT%\include simple_net.cpp %DNNLROOT%\lib\dnnl.lib
~~~

Refer to the
[Microsoft Visual Studio documentation](https://docs.microsoft.com/en-us/cpp/build/walkthrough-creating-and-using-a-dynamic-link-library-cpp?view=vs-2017)
on linking the application using MSVS solutions.

@note
Applications linked dynamically will resolve the dependencies at runtime.
Make sure that the dependencies are available in the standard locations
defined by the operating system or in the locations listed in the `PATH`
environment variable.
