# Adding Level Zero header to oneDNN sources

This RFC discusses the option of adding the Level Zero headers to the
oneDNN repository.

On Windows, Level Zero headers are not distributed as part of oneAPI
components. This incurs some extra work on the user side to manually
download Level Zero headers from GitHub and extract them (as described
in [oneDNN
documentation](https://docs.oneapi.io/versions/latest/onednn/dev_guide_build.html#id1)).

To simplify the process of building oneDNN DPC++ configuration on
Windows, this RFC proposes to include Level Zero headers in the oneDNN
repository.

## 1. Where to put them?
There are two options here:
- because we support Level Zero only for DPC++ configuration, it would
  make sense to add the headers in a `src/sycl/level_zero` directory
- the second option would be to group external components to a new
  `third_party` folder in the base directory of oneDNN. We currently
  have a number of external component that we regularly update
  (e.g. ittnotify, xbyak, xbyak\_aarch64), so it might be cleaner to
  just group them in a folder.

For simplicity, we propose to just go for the first option (add the
Level Zero headers to `src/sycl/level_zero`). We can at a later time
think about grouping third party programs in a separate folder.

## 2.When to update them?
Given that API is not broken from one minor release to the next, we
propose to update headers only when needed, typically if a feature was
added to Level Zero and we need to rely on it, or when a bug in
headers is fixed.

## 3. How to support multiple versions?
An open question is what to do when Level Zero releases its next major
release?  The main consideration here is that, as long as the
signature of the symbols oneDNN uses does not change, there is no
reason to update or support multiple versions of Level Zero
headers. This is because we use only the signatures from the headers,
and load symbols are runtime using dlsym/GetProcAddress.

If the signature of those symbols differ from Level Zero 1.x to 2.x,
there are mainly 2 options:
- support only one major version and bump the requirement for
  oneDNN. This is actually not really an option as it would bump the
  minimal requirements for oneDNN, and prevent many users from using
  oneDNN without updating their drivers/runtime.
- support multiple major versions of Level Zero. In this case, we can
  have separate sub-directories, for example `src/sycl/level_zero/v1`,
  `src/sycl/level_zero/v2`, ...

If we support multiple versions of Level Zero, we can either do it at
build time, or at runtime.

### 3.a Choose Level Zero version at build time

Here we would pick the right symbol signature statically at build
time. To achieve this, we would have to add a build knob to specify
the targeted Level Zero version (`DNNL_LEVEL_ZERO_VERSION`) which
would default to latest.
  
This option is simple, but it is rather restrictive for our users as
they will be able to target only one version of Level Zero, and this
will trickle down to their minimal requirements.

### 3.b Detect Level Zero runtime version at runtime
Currently, calls to Level Zero are actually wrapped in oneDNN (example
of
[zeKernelcreate](https://github.com/oneapi-src/oneDNN/blob/dd2abf12b06b0a82cebe531d3b2f440e71cadb64/src/sycl/level_zero_utils.cpp#L141-L152)). Here
we would add another level of wrappers, with one wrapper per major
version of Level Zero (e.g. `dnnl::zeKernelcreate_v1` and
`dnnl::zekernelcreate_v2`). The current wrappers
(e.g. `func_zeKernelCreate`) will then become dispatching function. In
order to dispatch, we would use the version of the Level Zero runtime,
which can be obtained from the SYCL device.  For this to work,
versioned wrappers will have to be compiled in separate object files,
and the headers with their declaration cannot contain any Level Zero
type (so all Level Zero objects will have to be passed as `void *`).

If Level Zero breaks API for the symbols we use, this RFC recommends
to go with the second option, which is to dynamically detect the Level
Zero runtime version, and dlSym/GetProcAddress according to the proper
signature for that version of the runtime.
  
# Summary
We propose to add Level Zero headers to a new `src/sycl/level_zero`
directory.  When Level Zero will release major releases, we will have
to assess if adding support for different versions is needed (if Level
Zero breaks API for the symbols we use). If they do, we will introduce
new sub-folders for each supported Level Zero version, and load Level
Zero symbols based on Level Zero runtime version.

