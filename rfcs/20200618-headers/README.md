# Changes in the oneDNN header install locations and namespace

## Introduction

New headers structure and namespaces are needed for cross-component and oneAPI
branding alignment.

The main concern is how to reach alignment while keeping backward
compatibility with existing code. The secondary concern is maintenance burden.

## Proposal

The following changes are to be implemented in the
[`dev-v2`](https://github.com/oneapi-src/onednn/tree/dev-v2) branch but will
eventually be part of master.

### Namespace changes

oneDNN should start providing two namespaces (C++ only): `dnnl::` and
`oneapi::dnnl::`. Either can be an alias. To minimize the changes, the
proposal is to alias the long one:

```c++
namespace dnnl { /* the usual stuff */ }

namespace oneapi { namespace dnnl = ::dnnl; }
```

The new namespace must be reflected in the
[spec](https://github.com/oneapi-src/oneAPI-spec/tree/master/source/elements/oneDNN).
There is no recommendation on whether longer or shorted namespace should be
used in the spec yet; this is outside of the scope of this RFC.

### Installation paths changes

#### Option 1 -- recommended

```
include
├── oneapi[/dnnl]
│   ├── dnnl_types.h
│   ├── dnnl_config.h
│   ├── dnnl.hpp
│   └── dnnl.h
├── dnnl_types.h        (#include "oneapi[/dnnl]/dnnl_types.h")
├── dnnl_config.h       (#include "oneapi[/dnnl]/dnnl_config.h")
├── dnnl.hpp            (#include "oneapi[/dnnl]/dnnl.hpp"")
└── dnnl.h              (#include "oneapi[/dnnl]/dnnl.h"")
```

The `/dnnl/` portion of the path may be present or not. The recommendation is
*to include it* so that it is possible to extend dnnl library later in a
straightforward manner.

Additionally, the compatibility headers can complain like this:

```c++
#ifdef _MSC_VER
#pragma message "Warning: this header is deprecated. Please include oneapi[/dnnl]/dnnl.hpp instead"
#else
#warning "This header is deprecated. Please include oneapi[/dnnl]/dnnl.hpp instead"
#endif

#include "oneapi[/dnnl]/dnnl.hpp"
```

The warnings can be suppressed via `-Wno-cpp` (clang, gcc),
`-diag-disable:1224` (icc). The MSVC message cannot be disabled, but it does
not show as a warning, just a message. (However, the internal oneDNN CI will
still treat them as warning because the word 'warning' appears in the
message.)

This option *is recommended*. The canonical include path to `dnnl.hpp` will
become `#include "oneapi/dnnl/dnnl.hpp"`.

#### Option 2 -- not recommended

```
include
├── oneapi[/dnnl]
│   ├── dnnl.hpp        (#include "..[/..]/dnnl.hpp")
├── dnnl_types.h
├── dnnl_config.h
├── dnnl.hpp
└── dnnl.h
```

The `include/dnnl.hpp` can complain if included directly similarly to option 1.

This option is *not* recommended because it only delays the inevitable and is
not as extensible as option 1. However, this can be a stop-gap measure for
beta08 release.

### Source code changes

Regardless of the install path changes, it would be necessary to:

- Change all examples and tests to `#include "oneapi[/dnnl]/dnnl.hpp"` so that
  they show the recommended way of using the library.

  * C include paths would have to be changed or not depending on whether
    option 1 or option 2 is chosen.

- Mimic the install structure in the repository `include/` directory for those
  who build oneDNN directly like TensorFlow.

### Future changes
The following header files can be removed in oneDNN 3.0 or later.

```
include
├── dnnl_types.h        (#include "oneapi/dnnl_types.h")
├── dnnl_config.h       (#include "oneapi/dnnl_config.h")
├── dnnl.hpp            (#include "oneapi/dnnl.hpp"")
└── dnnl.h              (#include "oneapi/dnnl.h"")
```

