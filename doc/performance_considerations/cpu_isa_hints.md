CPU ISA Hints {#dev_guide_cpu_isa_hints}
=======================================

For performance reasons, extra hints may be provided to oneDNN which enable the
just-in-time (JIT) code generation to prefer or avoid certain CPU ISA features.
An example is disabling Zmm vector registers usage in order to take advantage of
higher clock speed.

## Build-time Controls

At build-time, support for this feature is controlled via cmake option
`ONEDNN_ENABLE_CPU_ISA_HINTS`.

| CMake Option                | Supported values (defaults in bold) | Description           |
|:----------------------------|:------------------------------------|:----------------------|
| ONEDNN_ENABLE_CPU_ISA_HINTS | **ON**, OFF                         | Enables CPU ISA hints |

This results in making oneDNN aware of ONEDNN_CPU_ISA_HINTS environment variable
and corresponding setter utility namely, @ref dnnl::set_cpu_isa_hints routine.

## Run-time Controls

When the feature is enabled at build-time, the `ONEDNN_CPU_ISA_HINTS`
environment variable can be used to specify an ISA-specific hint to enable
oneDNN to dispatch an appropriate kernel.

| Environment variable | Value      | Description                                       |
|:---------------------|:-----------|:--------------------------------------------------|
| ONEDNN_CPU_ISA_HINTS | NO_HINTS   | Use default configuration for ISA                 |
| \                    | PREFER_YMM | Prefer to use YMM registers for vector operations |

This feature can also be managed at run-time with the following functions:

* @ref dnnl::set_cpu_isa_hints function allows changing the CPU ISA hint at
  run-time. The limitation is that it is possible to set the value only once. In
  addition, it is advised to call this function before any other oneDNN API.
  This is because the first internal query of CPU ISA hints will disable the
  ability to change it. Once disabled, changing the CPU ISA hint will return an
  error.
* @ref dnnl::get_cpu_isa_hints function returns the currently used CPU ISA hints
  and by default poses no restrictions.

Function settings take precedence over environment variables. Moreover, if the
hint is not applicable then it would be silently ignored. For instance with
SSE41 there are no YMM registers and so hint to prefer YMM registers would be
silently bypassed.
