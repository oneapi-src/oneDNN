Primitive Attributes: deterministic {#dev_guide_attributes_deterministic}
===================================================================

To support debugging, validation and certification of some applications, oneDNN
provides a deterministic mode. This mode guarantees that multiple executions of
the same primitive on a given platform return the exact same result bitwise.

For most primitives, oneDNN provides a run-to-run deterministic execution for a
fixed environment. In particular, if the hardware platform and software
environment (library/runtime versions, environment variables, etc.) are
identical between multiple runs, the produced results should be bit-wise
identical.

However, some implementations rely on non-deterministic constructs such as
atomic operations. In order to guarantee deterministic execution, a
deterministic attribute can be set (default false) with the
@ref dnnl_primitive_attr_set_deterministic (C API) or the
@ref dnnl::primitive_attr::set_deterministic (C++ API) functions.

The deterministic primitive attribute accepts:
- `false` (default): Permits the library to use non-deterministic constructs
      resulting in non-identical run-to-run outputs.
- `true`: Enforces dispatching of implementations with deterministic execution.

Enforcing deterministic execution might impact the performance of Convolution,
Matmul, and normalization primitives, especially on some GPU devices.
