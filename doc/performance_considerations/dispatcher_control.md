CPU dispatcher control {#dev_guide_cpu_dispatcher_control}
==========================================================

DNNL uses JIT code generation to implement most of its functionality and will
choose the best code based on detected processor features. Sometimes it is
necessary to control which features DNNL detects. This is sometimes useful for
debugging purposes or for performance exploration. To enable this, DNNL
provides two mechanisms: an environment variable `DNNL_MAX_CPU_ISA` and a
function `dnnl::set_max_cpu_isa()`.

The environment variable can be set to an upper-case name of the ISA as
defined by the `dnnl::cpu_isa` enumeration. For example,
`DNNL_MAX_CPU_ISA=AVX2` will instruct DNNL to never dispatch to JIT-ed CPU
primitive implementations that require ISA 'higher' than AVX2 like AVX512.
The `DNNL_MAX_CPU_ISA=FULL` setting implies no restrictions.

The `dnnl::set_max_cpu_isa()` function allows changing the ISA at run-time.
The limitation is that, it is possible to set the value only before the first
JIT-ed function is generated. This limitation ensures that the JIT-ed code
observe consistent CPU features both during generation and execution.

This feature can be enabled or disabled at build time. See @ref
dev_guide_build_options for more information.

Environment setting `DNNL_MAX_CPU_ISA=VANILLA` and `FULL` apply to both x86
and non-x86 systems.  The ordering of feature sets looks like:

VANILLA --> ANY --> cpu-specific ... --> FULL

- VANILLA is cross-platform code
- ANY use the most basic features (for completeness, useless for x86)
- FULL uses the fullest set of features
