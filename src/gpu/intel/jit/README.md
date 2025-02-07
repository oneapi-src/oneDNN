GPU JIT-compiled Kernels
===========================================

# Introduction

GPU JIT kernels utilize nGEN for emitting assembly code rather than coding them
in OpenCL C and compiling them down to assembly using the OpenCL runtime. The
main goals motivating the use of JIT development are:
- Assembly level performance
- Broad coverage including data types, propagation kind, HW, etc
- Generalization
    - Rely on common building blocks and optimizations
    - Avoid specialization unless it's absolutely necessary

# nGEN
nGEN is a library for generating Gen assembly, with the goal of producing
identical encodings to the Intel Graphics Assembler (IGA). For more information
on Gen assembly, see the [Programmer's Reference Manuals](https://www.intel.com/content/www/us/en/docs/graphics-for-linux/developer-reference/1-0/overview.html) and
[Introduction to Gen Assembly](https://www.intel.com/content/www/us/en/developer/articles/technical/introduction-to-gen-assembly.html).

Inspired by [Xbyak](https://github.com/herumi/xbyak), nGEN's syntax is meant to be as close to Gen assembly as
practical in C++. It can generate textual assembly, raw binary, and OpenCL
kernels (ready to be passed into `ndEnqueueNDRangeKernel`). Generation of OpenCL
kernels requires the OpenCL runtime.

nGEN uses generator classes to control the emission of instructions. oneDNN uses
the templated `generator_t<ngen::HW>` class for jit kernels, which is built off
of the `ngen::OpenCLCodeGenerator` class.

# Injectors
Injectors provide low-level interfaces for including common building block code
within kernels. They serve two main purposes:
- Provide consistent low-level optimizations across the library
- Remove some complexity, allowing for higher-level development

Injectors are written in pure nGEN so they can be used in any JIT kernels. For
an example of how to include injectors in IR code, see `jit/ir/eltwise.hpp`
and the corresponding functions in `jit/ir/codegen.cpp`.

# Intermediate Representation (IR)

oneDNN's IR adopts many ideas from the IR used by the [Halide](https://halide-lang.org/) project.

The IR is used to generate an Abstract Syntax Tree for each kernel, which can be
traversed and optimized via optimization passes once created. The additional
optimization potential from this strategy comes at the cost of more complexity
in the design of these kernels.
