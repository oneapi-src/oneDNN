# oneDNN interoperability API mockups

> See also: [MORE.md](MORE.md).

Here interoperability API means functions to
* create oneDNN objects from runtime-native objects,
* retrieve runtime-native objects from oneDNN objects,
* setting runtime-native objects underlying oneDNN objects.

The requirement to all the proposal is to keep the main `dnnl.hpp` free from
runtime-specific includes.

## Common API semantics update

### USM and SYCL buffers

Currently oneDNN requires a compile-time switch to enable USM. This should go
away. But there is still a question what a memory object constructed without a
handle should contain: a SYCL buffer or an USM pointer, and, if it is the
latter, whether the USM pointer should have a shared or device class.

* Option 1: a memory object constructed without a handle should contain a SYCL
  buffer. This option ensures oneDNN behavior is compatible between DPC++ and
  ComputeCPP which does not support USM (at least yet...). It also would work
  well regardless of whether the default queue is in or out of order. The
  downside of this option is that USM-based code will have to always provide
  pointers.

* Option 2a: a memory object constructed without handle should contain a
  device-specific USM pointer. This option makes SYCL GPU and SYCL CPU behave
  in fashion very similar to traditional CPU with respect to memory allocation.
  Alignment with respect to execution is only realized for in-order queues. The
  downside of this option is that buffers-based code will have to always use
  interoperability API and that out-of-order queues will require special
  handling.

* Option 2b: a memory object constructed without handle should contain a shared
  USM pointer. It may be an upgrade from 2a or, probably, the decision to
  allocate device-specific or shared pointer by default can be based on the
  capabilities of the HW underlying the engine.

## Runtime API options

*Option 1* introduces separate `dnnl::<runtime>` namespaces which contain all
the necessary interoperability functions. The interoperability functions are
unique to each runtime and are declared in separate header files.

*Option 2* adds interoperability functions as very generic templates that can
be later specialized inside the library.

*Option 3* and *Option 3a* implement interoperability in a way similar to
SYCL. Each runtime introduces separate header file with type traits mapping
oneDNN types to runtime-specific types, and the main `dnnl.hpp` file declares
interoperability functions templated with a runtime parameter.
