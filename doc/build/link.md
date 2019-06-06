Linking to the Library {#dev_guide_link}
===========================================

Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN)
includes several header files providing C and C++ APIs for the functionality
and one or several libraries depending on how Intel MKL-DNN was built.

## Header Files

| File                     | Description
| :---                     | :---
| include/mkldnn.h         | C header
| include/mkldnn.hpp       | C++ header
| include/mkldnn_types.h   | Auxiliary C header
| include/mkldnn_config.h  | Auxiliary C header
| include/mkldnn_version.h | C header with version information

## Libraries

### Linux

| File                  | Description
| :---                  | :---
| lib/libmkldnn.so      | Intel MKL-DNN dynamic library
| lib/libmkldnn.a       | Intel MKL-DNN static library (if built with `MKLDNN_LIBRARY_TYPE=STATIC`)

### macOS

| File                     | Description
| :---                     | :---
| lib/libmkldnn.dylib      | Intel MKL-DNN dynamic library
| lib/libmkldnn.a          | Intel MKL-DNN static library (if built with `MKLDNN_LIBRARY_TYPE=STATIC`)

### Windows

| File              | Description
| :---              | :---
| bin\libmkldnn.dll | Intel MKL-DNN dynamic library
| lib\libmkldnn.lib | Intel MKL-DNN import library

## Linking to Intel MKL-DNN

The examples below assume that Intel MKL-DNN is installed in the directory
defined in the `MKLDNNROOT` environment variable.

### Linux/macOS

~~~sh
g++ -std=c++11 -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib simple_net.cpp -lmkldnn
clang++ -std=c++11 -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib simple_net.cpp -lmkldnn
icpc -std=c++11 -I${MKLDNNROOT}/include -L${MKLDNNROOT}/lib simple_net.cpp -lmkldnn
~~~

@note
Applications linked dynamically will resolve the dependencies at runtime. Make
sure that the dependencies are available in the standard locations defined by
the operating system, in the locations listed in the `LD_LIBRARY_PATH` (Linux)
or `DYLD_LIBRARY_PATH` (macOS) environment variable or the `rpath` mechanism.

### Windows

To link the application from the command line, set up the `LIB` and `INCLUDE`
environment variables to point to the locations of the Intel MKL-DNN headers and
libraries.

~~~bat
icl /I%MKLDNNROOT%\include /Qstd=c++11 /qopenmp simple_net.cpp %MKLDNNROOT%\lib\mkldnn.lib
cl /I%MKLDNNROOT%\include simple_net.cpp %MKLDNNROOT%\lib\mkldnn.lib
~~~

Refer to the
[Microsoft Visual Studio documentation](https://docs.microsoft.com/en-us/cpp/build/walkthrough-creating-and-using-a-dynamic-link-library-cpp?view=vs-2017)
on linking the application using MSVS solutions.

@note
Applications linked dynamically will resolve the dependencies at runtime.
Make sure that the dependencies are available in the standard locations
defined by the operating system or in the locations listed in the `PATH`
environment variable.
