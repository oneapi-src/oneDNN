# RFC: Convert DNNL_VERBOSE output into benchdnn problems

## Motivation

There are several requests from users and developers to enable DNNL_VERBOSE
output be "runnable" from benchdnn. Such requests mostly consider the following
scenario: there is some integration of oneDNN into some application, and there
is some topology to run. If validation shows performance or correctness
regression, it is very hard and/or time consuming to narrow down the specific
guilty layer, and those users usually send a log file with DNNL_VERBOSE output,
which is massive. Manual conversion of these outputs into benchdnn problems
takes enormous amount of time, since the topology usually consists of 100+
layers. It is just inconvenient not to have some tool that allows to perform
such conversion very fast.

## Tool Requirements

* The tool should accept either a single line or a text file with lines.
* The output should be fully benchdnn-compatible.
* The tool should be easy to use.

## Proposal

### Option 1 (recommended) - Python script.

Use the power of Python to create such tool. It will take the input and produce
benchdnn batch files. Depending on the purpose, script may support different
modes, such as creating a single file to keep the order of operations, or
separate problems into different driver batch files, so that they could be run
(since benchdnn does not allow to work with several drivers simultaneously).
Script advantages are:
* Flexibility in terms of features support.
* Script defects will not affect a release schedule as it won't be a part of the
  product.
* The script will support current version of the library and benchdnn. But in
  case backward compatibility is required, it should not be hard to implement
  it.

### Option 2 - native benchdnn support.

Implement such support as a standalone benchdnn driver. All pros of a Python
script seems to be a cons of C++ solution:
* Each feature should be tested through regular oneDNN validation cycles,
  since it's a main validation tool.
* Errors and bugs in code will likely lead to patch fixes for a release.
* Versioning - earlier versions may lack of some suitable features.

## Testing

The idea behind advanced testing is to provide a batch file with various
problems, run them and obtain their verbose lines. Then produce a batch files
from these verbose lines and compare with the initial batch file. Such kind of
testing may require some features from benchdnn.

Simple testing could be just passing a verbose line and provide an expected
benchdnn command to compare with. This is likely a start point for testing,
though in future will require changes any time the expected output changes due
to incompatible updates.

EOD.
