# General Notes and Details

## Return status

Returns `1` if any submitted tests returned status `FAILED` or `UNIMPLEMENTED`,
`0` otherwise.

## Running Tests

oneDNN comes with its own testing infrastructure enabled through CMake. Tests
can be executed via the command:
``` sh
    make test_<test-name>
```
This instructs CMake to build a deployable project and run the specific test.

These tests target specific oneDNN features and are based on benchdnn
configurable executions.

The available tests can be found in the oneDNN directory:
tests/benchdnn/inputs/<primitive_name>/test_<test-name>.

## Glossary

| Abbreviation | Description
| :---         | :---
| src          | Source/input image
| wei          | Weights (or filter)
| bia          | Bias
| dst          | Destination/output image
| acc          | Accumulation (typically in terms of data type)

## Modes

**benchdnn** supports several execution flows ("modes"). Driver takes the
following steps to execute any flow:
1. Parse user input.
2. Iterate over multiple selected options for each problem descriptor and create
   a driver problem object for each setup. Each setup continues doing next
   steps.
3. Call backend API to create backend objects to execute.
4. Create memory objects for backend and reference paths and fill them with
   reasonable data.
5. Execute backend path.
6. Correctness validation:
   * Execute reference path.
   * Setup compare object.
   * Compare outputs of backend and reference.
   * Check that padded area, if any, is properly zeroed.
   * Report a test case status and repro line.
7. Performance validation:
   * If correctness was requested, proceed only if it passed.
   * Execute backend path in a loop until one of selected criterion to stop is
     triggered. Refer to [performance options](knobs_common.md) for details.
   * Print a performance report output based on selected options and collected
     statistics during the previous step.
8. Repeat steps 2-7 until all setups are validated.
9. Report the summary and return the status.

The following modes are supported:
* Correctness mode: This is the default driver flow. It executes steps above
  skipping step 7.
* Performance mode: This flow executes steps above skipping step 6.
* Correctness & performance mode: This flow executes all step above.
* Run mode: This flow executes steps 1-5 above. It allows to save time from
  running correctness when it is not needed. This mode is compatible with
  correctness or performance mode, though it will no longer be a run mode, but
  correctness or performance one.
* Listing mode: This flow executes steps 1-2 above. It allows to validate input
  files by parsing syntax and check if all problem repro lines are expected.
  This mode is standalone and is not compatible with other modes.

## Problem Statuses

Each problem in **benchdnn** has its status indicating the result of running a
problem in the correctness mode. Following statuses are supported:
* `EXECUTED`. It means that a problem was run but did not utilize correctness
  validation. This reflects that `Run` mode was used.
* `PASSED`. It means that a problem passed the validation, and a library output
  coincides with a reference path from the driver.
* `SKIPPED`. It means that a problem was not run and a brief reason is reported.
* `LISTED`. It means that a benchdnn problem was created and the reproducer line
  was reported. A primitive descriptor is not created in this case.
* `MISTRUSTED`. It means that correctness validation is invalid. This often
  happens when the result has more zeros than the threshold set for the number
  of zero values in the output. One possible reason is incorrect filling with
  input data for a given problem. Treated as `PASSED`.
* `FAILED`. It means that a problem did not pass the validation, and a library
  output differs from a reference path from the driver.
* `UNIMPLEMENTED`. It means that the library does not have an implementation for
  a requested problem. It is treated as `FAILED`.

## Input Files Naming Convention

Benchdnn follows certain [guidelines](benchdnn_input_files_naming_convention.md)
regarding input files naming convention.
