# Correctness Validation in Fast Reference Mode

## Introduction
To perform correctness validation **benchdnn** creates and executes a testing
object with specially-prepared memory objects. Then, a straightforward
reference kernel on `float` values is used to obtain the expected answer on the
data that is fed to the library. Next, the comparison between two outputs is
performed.

For the GPU backend, the reference kernel run on GPU memory objects may take a
significant amount of time. Because of this, the reference kernel is executed
on the CPU. But in most cases, the process will still be slow.

## Implementation and General Notes
To decrease the amount of time spent in reference kernels, benchdnn uses
libraries where highly optimized implementations are available. For example, a
CPU backend can be used for the GPU backend as their implementations are
independent of each other. The assumption is that the CPU backend is validated
independently against a driver reference code and is expected to provide the
same results as the GPU. The same idea is true for lower precision data types
for a CPU backend.

To speed up the reference kernel output, the driver will try to create a
second testing object depending on the original problem. Two supported
scenarios are GPU backend problems and CPU backend with reduced precision
floating-point data types (such as `bfloat16` or `float16`) problems. The
second testing object means the following:

* Create a `prb_t` object with identical settings the user requested besides
  memory formats. If the new testing object is successfully created, it will be
  used instead of the driver reference kernel. Identical data types are
  required to cover the quantized compute-bound primitives since their API
  support differs from the floating-point version.
* If the original data types do not allow the new CPU testing object
  to be created (for example, CPU ISA is limited to `AVX512_CORE` only, but
  `float16` was requested for GPU), the driver will try to create another one
  with `float` data types.
* If the latter testing object creation fails, then the driver reference kernel
  will be used.

## Debugging
Better performance comes with more complex debuggability. The above-mentioned
assumption does not always holds and issues may appear in CPU backend
implementations due to the variety of CPU architectures used on the systems
with the GPU. Additionally, correctness issues for the scenarios
the driver speeds up may not always come from the original testing objects but
from the additional ones. Lastly, issues observed on one system may not appear
on another one since systems may provide different core numbers and different
cache sizes, and both settings affect the blocking strategy for CPU testing
objects.

The reality brings the user to the following algorithm of debugging:

1. Try to reproduce the problem on a different system.
  - If reproduced as is, go to step 2.
  - If not reproduced, try to reproduce on the same system the issue was
    observed. If reproduced as is, go to step 2.
  - If neither option helps, an investigation is required.

2. To verify if the issue is coming from the original testing object, switch off
   the fast reference mode with `--fast-ref=false`.

3. Depending on step 2 outcome, several options are possible:
  - If the problem is still failing, this is the original testing object issue.
    Proceed with standard debug approaches.
  - If the problem stops failing, it may mean the CPU backend implementation
    has an issue. Try to reproduce it with `--engine=cpu`.
  - If reproduced, a tracker against the CPU implementation is appreciated.

Following the above steps will help to sort issues by buckets and should save
time spent on understanding where the issue originates. In general, issues
on the "reference" library object side should not happen often but the users
should not exclude that possibility.

## Limitations
Only certain drivers support faster reference implementations. They are limited
to conv, deconv, ip, and matmul. More or all drivers may be covered in the
future.

So far, the requirements for fast-ref feature are:

* Data filling for each tensor is independent from other tensors (which excludes
  normalization drivers).
