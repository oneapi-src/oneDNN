# Enable Clang-tidy for oneDNN
## Motivation
As the oneDNN team is growing it becomes harder to keep the codebase consistent.
The number of merge requests is increasing and there is usually quite a bit of code
review comments related to style violations or other easy-to-fix things, meaning that both
the merge request author and the reviewers have to spend time on fixing/reviewing those things.

## Proposal
The proposal is to enable a linter tool for oneDNN to automate checking for style violations
as well as for other easy-to-fix issues to improve codebase quality and simplify code review.
Clang-tidy is a linter tool that can help to automate addressing aforementioned issues.
Clang-tidy diagnoses typical programming errors like style violations, interface misuse,or bugs
that can be deduced via static analysis. Clang-tidy requires to build the library to perform
analysis. Therefore a git hook is not applicable here as it will be too expensive to
perform the analysis for each and every commit.

## Using Clang-tidy for oneDNN

### Integration With Cmake (preferable)
The build system provides `DNNL_USE_CLANG_TIDY` cmake option that has two possible values:

1. Check - enables checks from the clang-tidy configuration file
2. Fix - enables checks from the clang-tidy configuration file and instructs clang-tidy to fix
the hits if the [corresponding check](https://clang.llvm.org/extra/clang-tidy/checks/list.html) supports it

Prerequisites:

1. Cmake 3.6 or higher
2. Clang 9 or higher
3. Clang-tidy

### Use [run-clang-tidy](https://github.com/llvm-mirror/clang-tools-extra/blob/master/clang-tidy/tool/run-clang-tidy.py) or [clang-tidy-diff](https://github.com/llvm-mirror/clang-tools-extra/blob/master/clang-tidy/tool/clang-tidy-diff.py) scripts (optional)

Cmake 3.5 or higher supports `CMAKE_EXPORT_COMPILE_COMMANDS` option to generate `compile_commands.json`
that contains the exact compiler calls for all translation units of the project. The generated
compilation database is passed to the `run-clang-tidy` or `clang-tidy-diff` script with `-path`
option and the scripts start analyzing the codebase.

This way is considered optional as it requires more steps, however it's possible to implement a helper
script that would simplify use of those scripts, but the environment for Cmake and Clang compiler
must be set up prior running the helper script.

Prerequisites:

1. Cmake 3.5 or higher
2. Clang 9 or higher
3. Clang-tidy

## Enabling Clang-tidy in Regular Testing
It makes little sense to add support for clang-tidy in the build system, but not to enable it in
the regular testing.
The proposal is to enable it in CI as an additional build step. Since oneDNN doesn't have much OS
specific code then it would be sufficient to add the step for Linux only.

The clang-tidy hits are similar to compiler warnings except the fact that `DNNL_WERROR` has no effect
for these. In the case when the hits occur in CI there are two ways to fix those:
1. Go through all the warnings in the log file and fix them manually
2. Use `DNNL_USE_CLANG_TIDY=Fix` option to automatically fix them all. Note, that not all the
clang-tidy checks support the `fix` option. Also, sometimes, an auto-fix can lead to other issues that
most likely have to be fixed manually.

The clang-tidy step can be considered as a candidate for inclusion in the lightweight scan.

## Suppressing Undesired Diagnostics
Sometimes changing the code to suppress the warnings is not desired. For such situations clang-tidy
provides a way to suppress diagnostics with `NOLINT` and `NOLINTNEXTLINE` comments. The comments can be
used to disable all warnings or a particular one. A code snippet with an example can be found
[here](https://releases.llvm.org/9.0.0/tools/clang/tools/extra/docs/clang-tidy/index.html#suppressing-undesired-diagnostics).

## The List of Checks

Clang-tidy provides a long list of checks that are split into several [categories](http://clang.llvm.org/extra/clang-tidy/index.html#using-clang-tidy).
Below is a list of checks that seem to be reasonable for oneDNN. The list is subject to change.

### Readability
[readability-identifier-naming](https://clang.llvm.org/extra/clang-tidy/checks/readability-identifier-naming.html#readability-identifier-naming)

* readability-identifier-naming.ClassCase = lower_case
* readability-identifier-naming.StructCase = lower_case
* readability-identifier-naming.ClassSuffix = _t
* readability-identifier-naming.StructSuffix = _t

[readability-const-return-type](https://clang.llvm.org/extra/clang-tidy/checks/readability-const-return-type.html#readability-const-return-type)

[readability-redundant-smartptr-get](https://clang.llvm.org/extra/clang-tidy/checks/readability-redundant-smartptr-get.html#readability-redundant-smartptr-get)

[readability-misleading-indentation](https://clang.llvm.org/extra/clang-tidy/checks/readability-misleading-indentation.html#readability-misleading-indentation)

[readability-redundant-control-flow](https://clang.llvm.org/extra/clang-tidy/checks/readability-redundant-control-flow.html#readability-redundant-control-flow)

[readability-redundant-member-init](https://clang.llvm.org/extra/clang-tidy/checks/readability-redundant-member-init.html#readability-redundant-member-init)

[readability-redundant-string-cstr](https://clang.llvm.org/extra/clang-tidy/checks/readability-redundant-string-cstr.html#readability-redundant-string-cstr)

[readability-redundant-string-init](https://clang.llvm.org/extra/clang-tidy/checks/readability-redundant-string-init.html#readability-redundant-string-init)

[readability-simplify-subscript-expr](https://clang.llvm.org/extra/clang-tidy/checks/readability-simplify-subscript-expr.html#readability-simplify-subscript-expr)

[readability-static-accessed-through-instance](https://clang.llvm.org/extra/clang-tidy/checks/readability-static-accessed-through-instance.html#readability-static-accessed-through-instance)

[readability-static-definition-in-anonymous-namespace](https://clang.llvm.org/extra/clang-tidy/checks/readability-static-definition-in-anonymous-namespace.html#readability-static-definition-in-anonymous-namespace)

[readability-uniqueptr-delete-release](https://clang.llvm.org/extra/clang-tidy/checks/readability-uniqueptr-delete-release.html#readability-uniqueptr-delete-release)

[readability-container-size-empty](https://clang.llvm.org/extra/clang-tidy/checks/readability-container-size-empty.html#readability-container-size-empty)

[readability-delete-null-pointer](https://clang.llvm.org/extra/clang-tidy/checks/readability-delete-null-pointer.html#readability-delete-null-pointer)

[readability-else-after-return](https://clang.llvm.org/extra/clang-tidy/checks/readability-else-after-return.html#readability-else-after-return)


### Performance
[performance-for-range-copy](https://clang.llvm.org/extra/clang-tidy/checks/performance-for-range-copy.html#performance-for-range-copy)

[performance-implicit-conversion-in-loop](https://clang.llvm.org/extra/clang-tidy/checks/performance-implicit-conversion-in-loop.html#performance-implicit-conversion-in-loop)

[performance-inefficient-algorithm](https://clang.llvm.org/extra/clang-tidy/checks/performance-inefficient-algorithm.html#performance-inefficient-algorithm)

[performance-inefficient-string-concatenation](https://clang.llvm.org/extra/clang-tidy/checks/performance-inefficient-string-concatenation.html#performance-inefficient-string-concatenation)

[performance-inefficient-vector-operation](https://clang.llvm.org/extra/clang-tidy/checks/performance-inefficient-vector-operation.html#performance-inefficient-vector-operation)

[performance-move-const-arg](https://clang.llvm.org/extra/clang-tidy/checks/performance-move-const-arg.html#performance-move-const-arg)

[performance-unnecessary-copy-initialization](https://clang.llvm.org/extra/clang-tidy/checks/performance-unnecessary-copy-initialization.html#performance-unnecessary-copy-initialization)

[performance-unnecessary-value-param](https://clang.llvm.org/extra/clang-tidy/checks/performance-unnecessary-value-param.html#performance-unnecessary-value-param)

### Modernize
[modernize-make-shared](https://clang.llvm.org/extra/clang-tidy/checks/modernize-make-shared.html#modernize-make-shared)

[modernize-use-bool-literals](https://clang.llvm.org/extra/clang-tidy/checks/modernize-use-bool-literals.html#modernize-use-bool-literals)

[modernize-use-emplace](https://clang.llvm.org/extra/clang-tidy/checks/modernize-use-emplace.html#modernize-use-emplace)

[modernize-use-equals-default](https://clang.llvm.org/extra/clang-tidy/checks/modernize-use-equals-default.html#modernize-use-equals-default)

[modernize-use-override](https://clang.llvm.org/extra/clang-tidy/checks/modernize-use-override.html#modernize-use-override)

[modernize-use-nullptr](https://clang.llvm.org/extra/clang-tidy/checks/modernize-use-nullptr.html#modernize-use-nullptr)

[modernize-use-using](https://clang.llvm.org/extra/clang-tidy/checks/modernize-use-using.html#modernize-use-using)

### Bugprone
[bugprone-assert-side-effect](https://clang.llvm.org/extra/clang-tidy/checks/bugprone-assert-side-effect.html#bugprone-assert-side-effect)

[bugprone-copy-constructor-init](https://clang.llvm.org/extra/clang-tidy/checks/bugprone-copy-constructor-init.html#bugprone-copy-constructor-init)

[bugprone-forward-declaration-namespace](https://clang.llvm.org/extra/clang-tidy/checks/bugprone-forward-declaration-namespace.html#bugprone-forward-declaration-namespace)

[bugprone-move-forwarding-reference](https://clang.llvm.org/extra/clang-tidy/checks/bugprone-move-forwarding-reference.html#bugprone-move-forwarding-reference)

[bugprone-parent-virtual-call](https://clang.llvm.org/extra/clang-tidy/checks/bugprone-parent-virtual-call.html#bugprone-parent-virtual-call)

[bugprone-too-small-loop-variable](https://clang.llvm.org/extra/clang-tidy/checks/bugprone-too-small-loop-variable.html#bugprone-too-small-loop-variable)

[bugprone-undefined-memory-manipulation](https://clang.llvm.org/extra/clang-tidy/checks/bugprone-undefined-memory-manipulation.html#bugprone-undefined-memory-manipulation)

[bugprone-unhandled-self-assignment](https://clang.llvm.org/extra/clang-tidy/checks/bugprone-unhandled-self-assignment.html#bugprone-unhandled-self-assignment)

[bugprone-multiple-statement-macro](https://clang.llvm.org/extra/clang-tidy/checks/bugprone-multiple-statement-macro.html#bugprone-multiple-statement-macro)

[bugprone-macro-parentheses](https://clang.llvm.org/extra/clang-tidy/checks/bugprone-macro-parentheses.html#bugprone-macro-parentheses)

### Google Style
[google-default-arguments](https://clang.llvm.org/extra/clang-tidy/checks/google-default-arguments.html#google-default-arguments)

### Miscellaneous
[misc-misplaced-const](https://clang.llvm.org/extra/clang-tidy/checks/misc-misplaced-const.html#misc-misplaced-const)

[misc-definitions-in-headers](https://clang.llvm.org/extra/clang-tidy/checks/misc-definitions-in-headers.html#misc-definitions-in-headers)

[misc-redundant-expression](https://clang.llvm.org/extra/clang-tidy/checks/misc-redundant-expression.html#misc-redundant-expression)

[misc-uniqueptr-reset-release](https://clang.llvm.org/extra/clang-tidy/checks/misc-uniqueptr-reset-release.html#misc-uniqueptr-reset-release)

[misc-unused-alias-decls](https://clang.llvm.org/extra/clang-tidy/checks/misc-unused-alias-decls.html#misc-unused-alias-decls)

[misc-unused-using-decls](https://clang.llvm.org/extra/clang-tidy/checks/misc-unused-using-decls.html#misc-unused-using-decls)


## Clang-tidy Configuration File
The configuration file named `.clang-tidy` is located in the project root directory.
The snippet below shows a simple example of the configuration file.
```
Checks: >
    -*,
    readability-identifier-naming

CheckOptions:
  - key:             readability-identifier-naming.StructSuffix
    value:           _t
```

