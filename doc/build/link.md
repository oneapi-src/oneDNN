Linking to the Library {#dev_guide_link}
===========================================

oneDNN includes several header files providing C and C++ APIs
for the functionality and one or several libraries depending
on how oneDNN was built.

## Header Files

| File                                         | Description                        |
|:---------------------------------------------|:-----------------------------------|
| ``include/oneapi/dnnl/dnnl.h``               | C header                           |
| ``include/oneapi/dnnl/dnnl.hpp``             | C++ header                         |
| ``include/oneapi/dnnl/dnnl_types.h``         | Auxiliary C header                 |
| ``include/oneapi/dnnl/dnnl_config.h``        | Auxiliary C header                 |
| ``include/oneapi/dnnl/dnnl_version.h``       | C header with version information  |
| ``include/oneapi/dnnl/dnnl_graph.h``         | C header for graph API             |
| ``include/oneapi/dnnl/dnnl_graph.hpp``       | C++ header for graph API           |
| ``include/oneapi/dnnl/dnnl_graph_types.h``   | Auxiliary C header for graph API   |
| ``include/oneapi/dnnl/dnnl_ukernel.h``       | C header for ukernel API           |
| ``include/oneapi/dnnl/dnnl_ukernel.hpp``     | C++ header for ukernel API         |
| ``include/oneapi/dnnl/dnnl_ukernel_types.h`` | Auxiliary C header for ukernel API |

## Libraries

### Linux

| File           | Description                                                      |
|:---------------|:-----------------------------------------------------------------|
| lib/libdnnl.so | oneDNN dynamic library                                           |
| lib/libdnnl.a  | oneDNN static library (if built with `DNNL_LIBRARY_TYPE=STATIC`) |

### macOS

| File              | Description                                                      |
|:------------------|:-----------------------------------------------------------------|
| lib/libdnnl.dylib | oneDNN dynamic library                                           |
| lib/libdnnl.a     | oneDNN static library (if built with `DNNL_LIBRARY_TYPE=STATIC`) |

### Windows

| File         | Description                                                                                |
|:-------------|:-------------------------------------------------------------------------------------------|
| bin\dnnl.dll | oneDNN dynamic library                                                                     |
| lib\dnnl.lib | oneDNN import or full static library (the latter if built with `DNNL_LIBRARY_TYPE=STATIC`) |

## Linking to oneDNN

The examples below assume that oneDNN is installed in the directory
defined in the `DNNLROOT` environment variable.

### Linux/macOS

~~~sh
g++ -I${DNNLROOT}/include -L${DNNLROOT}/lib getting_started.cpp -ldnnl
clang++ -I${DNNLROOT}/include -L${DNNLROOT}/lib getting_started.cpp -ldnnl
icpx -I${DNNLROOT}/include -L${DNNLROOT}/lib getting_started.cpp -ldnnl
~~~

@note
Applications linked dynamically will resolve the dependencies at runtime. Make
sure that the dependencies are available in the standard locations defined by
the operating system, in the locations listed in the `LD_LIBRARY_PATH` (Linux)
or `DYLD_LIBRARY_PATH` (macOS) environment variable or the `rpath` mechanism.

#### Support for macOS hardened runtime

oneDNN requires the
[com.apple.security.cs.allow-jit](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_security_cs_allow-jit)
entitlement when it is integrated with an application that uses the macOS
[hardened runtime](https://developer.apple.com/documentation/security/hardened_runtime_entitlements).
This requirement comes from the fact that oneDNN generates code on
the fly and then executes it.

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
    <key>com.apple.security.cs.allow-jit</key><true/>
</dict>
</plist>
~~~

### Windows

The examples below assume that oneDNN is installed in the directory
defined in the `DNNLROOT` environment variable.

~~~bat
icx /EHa /I"%DNNLROOT%\include" getting_started.cpp "%DNNLROOT%\lib\dnnl.lib"
cl /EHa /I"%DNNLROOT%\include" getting_started.cpp "%DNNLROOT%\lib\dnnl.lib"
~~~

@note
You may also add paths to oneDNN headers and libraries to `LIB` and `INCLUDE`
environment variables instead of specifying these in the build command.

Refer to the
[Microsoft Visual Studio documentation](https://docs.microsoft.com/en-us/cpp/build/walkthrough-creating-and-using-a-dynamic-link-library-cpp?view=vs-2017)
on linking the application using MSVS solutions.

@note
Applications linked dynamically will resolve the dependencies at runtime.
Make sure that the dependencies are available in the standard locations
defined by the operating system or in the locations listed in the `PATH`
environment variable.
