# RFC: Build Graph API By Default

## Background

oneDNN Graph API was proposed and implemented on oneDNN master as an
experimental feature in October, 2022. Since then,

- The Graph API was released as an experimental feature in oneDNN v3.0.
- The Graph API specification was published in oneAPI Specification v1.2.
- The Graph API and implementation has been validated in oneDNN validation
  system covering both CPU and GPU platforms.
- A new benchdnn graph driver was designed and implemented to support
  correctness and performance benchmark for the Graph API.
- Frontend frameworks have been migrating the Graph API integrations from the
  legacy dev-graph branch to oneDNN master and v3.0 release.

As an experimental feature, the Graph API and implementation depend on a new
build option `ONEDNN_BUILD_GRAPH` which is OFF by default in oneDNN's build
system. The users have to turn it ON explicitly in the CMake line to access
Graph API features.

This RFC is proposing to enable and build Graph API and its implementation by
default in oneDNN's build system. With that, the oneDNN Graph API feature turns
from experimental to general available in the next release.

## Proposals

### `ONEDNN_BUILD_GRAPH`

Though it's proposed to enable and build Graph API by default in oneDNN's build
system, we will still keep and expose the build option `ONEDNN_BUILD_GRAPH`. The
proposal is to change the default value of the option from `OFF` to `ON` in
oneDNN CMake options. The reasons are:

- With keeping `ONEDNN_BUILD_GRAPH`, we also keep the backward compatibility if
  the users have explicit settings for the option in their build configurations.
- If the users do not want to build and use Graph API in their projects, they
  can still disable it through setting `ONEDNN_BUILD_GRAPH=OFF`.

The corresponding build option document will be changed accordingly.

### Work with other build options

As described in [build options
document](https://oneapi-src.github.io/oneDNN/dev_guide_build_options.html#graph-component-limitations),
Graph API and the build option `ONEDNN_BUILD_GRAPH` do not work with some values
of other build options. Specifying the options and values simultaneously in one
build will lead to a CMake error. Turning `ONEDNN_BUILD_GRAPH` ON by default
will affect the users who are using the build options and values listed in the
document (e.g., OpenVINO). To address the issue, we have below options:

#### Option 1: CMake error

oneDNN build system will throw CMake errors to users when incompatible build
options are set, for example, both `ONEDNN_BUILD_GRAPH=ON` and
`ONEDNN_GPU_RUNTIME=OCL` are set in the CMake line. The users are responsible to
fix the CMake error by changing either `ONEDNN_BUILD_GRAPH` to OFF or
`ONEDNN_GPU_RUNTIME` to other values. This can be considered as a breaking
change to related downstream projects.

#### Option 2: CMake warning

oneDNN build system will change to prompt CMake warnings to users when
incompatible build options are set. Underneath, the build system will turn OFF
`ONEDNN_BUILD_GRAPH` implicitly and execute the build process. The resulted
binary will not contain any Graph API and features.

Take the same example as above, the warning message can read as "Warning:
ONEDNN_BUILD_GRAPH=ON and ONEDNN_GPU_RUNTIME=OCL cannot work together. oneDNN is
building the library with ONEDNN_BUILD_GRAPH=OFF.".

With this option, we will keep backward compatibility for those downstream
projects which are using build options which are incompatible to the Graph API.

The problems of this option are:

- CMake warning messages are usually less visible to users and ignored.
- Silent success and the resulted binary may cause confusion to users.
- We will have to maintain the weird build logic for a long time.

#### Option 3: CMake warning + deprecation message

This option is more like a combination of option 1 and option 2. CMake warnings
will be prompted to users and the library will build implicitly with
`ONEDNN_BUILD_GRAPH=OFF`. The warning message will contain additional
deprecation message about that the build behavior will be changed in the future
releases and the users are encouraged to fix their build command lines before
the change happens.

Take the same example as above, the warning message can read as "Warning:
ONEDNN_BUILD_GRAPH=ON and ONEDNN_GPU_RUNTIME=OCL cannot work together. oneDNN is
building the library with ONEDNN_BUILD_GRAPH=OFF. This will turn to be a CMake
error in the next release.".

With this option, we will not break downstream projects immediately by giving
them enough time to fix the build system. But the downside is similar as option
2 as we will still have the silent success and confusion issues in a recent
release.

#### Decision

The decision is go with option 1.

### API and implementation

The changes will only impact the build system and will keep oneDNN API and
implementation unchanged.

### Validation

`ONEDNN_BUILD_GRAPH` is specified to ON or OFF explicitly for different jobs in
the internal validation system. The proposed changes will not affect the
validation behavior of the jobs.

### Release

It's proposed to implement the changes on oneDNN master branch and release them
in oneDNN v3.1 as a general available feature.

(EOD)
