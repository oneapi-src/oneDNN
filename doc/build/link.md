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

#### Support for macOS hardened runtime

DNNL requires the
[com.apple.security.cs.allow-unsigned-executable-memory](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_security_cs_allow-unsigned-executable-memory)
entitlement when it is integrated with an application that uses the macOS
[hardened runtime](https://developer.apple.com/documentation/security/hardened_runtime_entitlements).
This requirement comes from the fact that DNNL generates executable code on
the fly and does not sign it.

It can be enabled in Xcode or passed to `codesign` like this:
~~~sh
codesign -s "Your identity" --options runtime --entitlements Entitlements.plist [other options...] /path/to/libdnnl.dylib
~~~

Example `Entitlements.plist`:
~~~xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.cs.allow-unsigned-executable-memory</key><true/>
</dict>
</plist>
~~~

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
