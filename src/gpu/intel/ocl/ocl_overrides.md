## Overrides

A few flags can be set at runtime (dev mode only) to affect the compilation of OpenCL kernels, to help in debugging. To use them, set the following environment variables during execution:

- enable_ocl_werror (values: 0/1): Set to 1 to add the `-Werror` flag to kernel compilations.
- enable_check_assumptions (values: 0/1): Set to 1 to convert compiler assumptions (using the `ASSUME` macro) into runtime checks to confirm their validity.
- ocl_debug (values: 0/1): Set to 1 to enable debug prints

## Debuginfo verbosity levels

Set `ONEDNN_VERBOSE=debuginfo=X` where `X` is in one of the ranges below, to get additional information regarding compilation of OpenCL kernels:

- \>=5: Dump options and defines passed while building the kernel to the terminal.
- \>=10: Dump preprocessed kernel source code to the terminal. Due to slight differences in the cpp and opencl proprocessors, this may differ slightly from the actual kernel.