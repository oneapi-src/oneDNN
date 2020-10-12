# Update benchdnn input files for CI validation

## Introduction

Since amount of oneDNN functionality grows along with the necessary amount of
testing coverage, CI testing time increases rapidly. In order to have meaningful
functional testing with benchdnn, checking that it works and basic functionality
is not broken for most popular compilers when new changes are introduced, there
is a desire to separate general usability cases and implementation specific
cases targeted performance (in terms of ISA branches, different memory formats,
etc.) and move latter to the next level testing leaving general usability cases
most frequently tested.

In addition to this, current public CI, available for oneDNN external
contributors, does not cover correctness testing properly since benchdnn
validation is absent there. This proposal targets to close this gap as well.

For internal oneDNN testing infrastructure re-considered and shortened CI
coverage is expected to decrease machines load leading to higher testing cycles
throughput.

Current testing run time takes from 45 minutes and up to 20 hours, depending
on various factors, such as OS, ISA, Engine and infrastructure stability and
availability.

## Proposal

### Top level enabling bullets

- CI inputs to cover HW-agnostic features meaning most library API abilities on
  several relatively small shapes for all drivers.

- Stop separating input files for CPU/GPU backend - library claims to provide
  aligned support on both. All internal knowledge about unimplemented cases will
  be hidden in benchdnn internals (like it's done for Gtests currently).

- oneDNN team is responsible to test CI inputs to secure there are no breaking
  changes to all backends it supports.

- If a dedicated input is desired for any backend and it can't be re-used with
  existing input files, its enabling should follow documented rules for naming
  convention and CMake target enabling mechanism.

### Implementation bullets

- Get rid of `--allow-unimpl` flag support to avoid potential leak of functional
  support. All known unsupported cases to be covered in benchdnn sources. Such
  problems will be marked as SKIPPED and provide a short description explaining
  the skipping reason. [The change is already in master.]

- Move all benchdnn inputs to a common nomenclature: test, harness, option_set,
  set and shapes files. Uniform and update documentation what each entity means
  and what restrictions apply. [The update is already in master.]

- Move most shapes functional testing to Nightly testing and uniform the
  coverage between CPU and GPU backends to avoid bug leaks.

- Build up a feature of testing all available implementations in the library for
  a given problem, since x64 implementations can't be used for non-x64 builds.
  This targets to cover compiler-optimized implementations (known as `simple`).

## Proposed coverage

New file names: test_$driver_ci and shapes_ci

* --dir=all # everything that is applicable for a driver, including FWD_I if ws is used.

* --{s,d,}dt=all # everything that is applicable for a driver

* --cfg=f32,bf16bf16bf16,f16,u8s8u8,s8s8s32 # integer flavors with different dst data type

* --{s,d,stat,}tag=any,abx,axb # only plain formats to cover + `any` where applicable

* --alg=all # everything applicable

* --attr=all # one case for each supported combination where applicable + empty

* --inplace=true,false # where applied

* --skip-impl= # no skipping in ci

* --mb=2

* --axis=0,1 # where applicable

* --$driver-specific-options-to-cover-api # bnorm, lnorm, matmul, rnn, shuffle

* --runtime-options # matmul and reorder

Problems to run:
* Conv: shapes_basic, which contains 6 simple cases + tailed modifications for
  each of those.

* Rest: simple simd-friendly and tailed cases for 0D - 3D spatial cases.
    - Binary: additionally, to cover broadcast feature.

Goal: to fit benchdnn testing in 10-20 minutes total.

EOD
