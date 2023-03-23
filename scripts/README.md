# Scripts

## Generating debug header

`generate_dnnl_debug.py` generates the oneDNN debug header and source files with
enum to string mapping. Each time a new tag is added to the API, this script
should be executed to re-generate the debug header and relevant source code.

### Usage

```sh
# Generate dnnl_config.h
# -DDNNL_EXPERIMENTAL_SPARSE=ON is required to preserve sparse-specific symbols
$ (mkdir -p build && cd build && cmake -DONEDNN_BUILD_GRAPH=OFF -DDNNL_EXPERIMENTAL_SPARSE=ON ..)

# Generate types.xml
# CastXML can be found at https://github.com/CastXML/CastXML
$ castxml --castxml-cc-gnu-c clang --castxml-output=1 -Iinclude -Ibuild/include include/oneapi/dnnl/dnnl_types.h -o types.xml

# run generate_dnnl_debug.py
$ ./scripts/generate_dnnl_debug.py types.xml
```


## Generating format tags

`generate_format_tags.py` generates C++ API tags based on C API tags. Each time
a new tag is added to the C API, this script should be executed to add this tag to the
C++ API.

### Usage

```sh
$ ./scripts/generate_format_tags.py
```


## Verbose converter

See [verbose_converter/README.md](verbose_converter/README.md)
