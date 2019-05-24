Profiling with Intel(R) VTune Amplifier {#dev_guide_vtune}
========================================================

Intel MKL-DNN uses just-in-time compilation (JIT) to generate optimal code
for some functions based on input parameters and instruction set supported
by the system. The library uses Intel SEAPI interface to annotate the
generated code for Intel VTune Amplifier, so that it can correctly attribute
processor monitoring unit (PMU) events to the code.

The behavior is controlled with build time option `MKLDNN_ENABLE_JIT_PROFILING`

| Option                      | Possible Values (defaults in bold)   | Description
| :---                        |:---                                  | :---
|MKLDNN_ENABLE_JIT_PROFILING  | **ON**, OFF                          | Enables integration with Intel(R) VTune(TM) Amplifier
