CPU dispatcher control {#dev_guide_cpu_dispatcher_control}
==========================================================

oneDNN uses JIT code generation to implement most of its functionality and will
choose the best code based on detected processor features. Sometimes it is
necessary to control which features oneDNN detects. This is sometimes useful for
debugging purposes or for performance exploration. To enable this, oneDNN
provides two mechanisms: an environment variable `DNNL_MAX_CPU_ISA` and a
function `dnnl::set_max_cpu_isa()`.

The environment variable can be set to an upper-case name of the ISA as
defined by the `dnnl::cpu_isa` enumeration. For example,
`DNNL_MAX_CPU_ISA=AVX2` will instruct oneDNN to dispatch code that will run
on systems with Intel AVX2 instruction set support. The `DNNL_MAX_CPU_ISA=ALL`
setting implies no restrictions.

The `dnnl::set_max_cpu_isa()` function allows changing the ISA at run-time.
The limitation is that, it is possible to set the value only before the first
JIT-ed function is generated. This limitation ensures that the JIT-ed code
observe consistent CPU features both during generation and execution.

This feature can be enabled or disabled at build time. See @ref
dev_guide_build_options for more information.
