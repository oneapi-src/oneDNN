Proposal to improve troubleshooting and debugging in oneDNN

# 1. Introduction

Customers often voice their difficulties troubleshooting the issues
they face when using oneDNN. The most common issues are:
- when a function call is not successful, they get no actionable
  information. For example, if a primitive descriptor creation fails,
  they would most of the time get an `invalid_argument` status, which
  is not helpful to know which argument is invalid, or in the case of
  inconsistencies between several arguments dimension, no hint is
  given to them to correct their issues.
- when running a primitive, sometimes we don't dispatch the most
  efficient implementation. The issue here is that a user does not
  know how they could change their parameters to get more efficient
  implementations, as we don't give any hint on why a given
  implementation is not dispatched.

Currently, the only options for a user is to either dig inside oneDNN
code where the `invalid_argument` or `unimplemented` statuses were
raised, or to submit an issue against the oneDNN team for
investigation.


On the other hand, it also seems that debug printing is necessary for
developers. It seems a few implementations already rely on their own
mechanism to print dumps or implementation specific information.

The current proposal aims at developing an internal framework to
simplify troubleshooting for users, and also improve debuggability for
developers, as there seem to be a need for both.


# 2. Improve verbose output, the aspects to consider

## 2.1 Library statuses
We can think about extending our statuses in order to provide more
granularity.
  
When dealing with C++ API, we can safely add new exceptions by
inheriting from the existing ones.  This has the benefit of not
breaking existing codes/semantics, while at the same time it provides
more granularity in error messaging to the user.
However, this is trickier for the C API. In that case, if we break
down a status into multiple statuses (e.g. instead of returning
`invalid_argument`, we return `invalid_argument_X`), then it becomes
hardly programmable if a user wants to catch all those.


Another aspect is that it does not allow to help our user for
non-failing diagnostics. For example, dispatching information cannot
be obtained from status, as:
- the primitive descriptor was created successfully, even though the
  dispatched implementation is not the one expected by the user.
- each implementation that was not dispatched might have a different
  reason for not dispatching

For the reasons above, we propose to leave the statuses as they
currently are, so that they align with std::exception, and can be
easily mapped from C to C++ API.


## 2.2 Release mode verbose vs debug mode verbose
Some debugging information might impact the performance of the
library, in particular primitive creation and execution time.
Currently, we already have such debugging facilities all over the
place, and they are mostly controlled with dedicated macros, that are
implementation specific (to name a few, we have `GEN_CONV_DEBUG`,
`TR_DEBUG`, `DEBUGPRINT`).

Here we propose to unify those macros. The main benefit is, for
developers, that the mechanism to print and debug would be the same
across implementations.

This macro should be independent of `NDEBUG` as we might need to use
this logging information on the release build. We propose to call this
unified macro `ONEDNN_DEVEL_MODE`, and it should have a matching cmake
option.

We recommend that all messages related to internal logic go under
`ONEDNN_DEVEL_MODE`. Only messages related to user visible APIs which
are actionable by user should live outside `ONEDNN_DEVEL_MODE`.

## 2.3 Verbose levels vs verbose flags

Currently, we have some documented information messages we provide
(primitive creation/execution time), and some undocumented
(e.g. runtimes info and implementation details).  For the undocumented
ones, we don't have alignment on which level they should print (some
print at level 2 and above, other print at level 5 and above).

The current verbose levels were designed with the assumption that each
level of information was inclusive of the previous ones. If we want to
increase the number of categories of information to print
(e.g. dispatch info, perf, error info, debug info, ...), it will
quickly bloat the verbose output, which can become an issue when using
verbose for long running workloads.  More importantly, verbose
messages might impact runtime (e.g. getting perf on GPU), so getting
some information sometimes impacts runtime.

We propose here to use flags in order to limit verbose bloat, and
allow printing messages without impacting runtime when possible. This
could take two forms:
- a list of message types, like `ERROR,PROFILE,DISPATCH,DEVINFO`...
- a bit mask encoded in an integer, -1 being all messages are
  displayed. So we would have for example ERROR=1, INFO=2, DISPATCH=4,
  ... and the user would have to pass the sum of all diagnostics they
  want (e.g. for `ERROR` and `DISPATCH`, they would have to pass
  `ONEDNN_VERBOSE=5`).

We would recommend going with the first option as it is simpler on the
user side. It would make parsing ONEDNN_VERBOSE more involved, but the
overhead should be negligible as it happens only once upon printing
the header of the verbose log.

Regarding the categories of information to print, we would recommend the following:
- `ERROR`, for all error messages. It can be for runtime / third-party
  components errors, or any library entry-point that return an error
  status.
- `DISPATCH`, for all information related to dispatching
  information. This is mostly to print why each implementation
  eligible for dispatch did not dispatch.
- `PROFILE`, `PROFILE_EXEC` and `PROFILE_CREATE`, for all messages
  providing performance of primitive creation and execution.
- `ALL`, would display all messages.


To map the current verbose levels we have, we would have to expose
`ERROR`, `PROFILE` and `DEVINFO`.

So the equivalent of current level 1 would be
`ONEDNN_VERBOSE=ERROR,PROFILE_EXEC` and the equivalent of our
current level 2 would be `ONEDNN_VERBOSE=ERROR,PROFILE` or
`ONEDNN_VERBOSE=ALL`.

To not disturb the current users, we would still accept 0, 1 and 2 and
map them internally to their new equivalent.

## 2.4 Choosing a uniform verbose format

Currently, each implementation is responsible for formatting verbose
messages.  This leads to non uniform format and verbose levels for a
given information.

For example, if we take error reporting in library, we have the
following line headers (note the different positions for device type
and component as well as the different error labels).
- `onednn_verbose,error,<pd_info>`
- `onednn_verbose,gpu,error,...`, note that this one is also used for
  some ocl and ze errors.
- `onednn_verbose,error,gpu,...`
- `onednn_verbose,gpu,ocl_error,...`
- `onednn_verbose,gpu,ze_error,...`
- `onednn_verbose,gpu,sycl_exception`
- `onednn_verbose,cpu,error,acl,...`
- `onednn_verbose,jit_perf,error,...`

We propose the following common message "header" format
`onednn_verbose,[<stamp>,]<msg_type>,<component>,`

with:
- `msg_type` one of `error`, `info`, `profile_exec`,...
- `component` one of `common`, `cpu`, `gpu`, `ocl`, `sycl`,
  `level_zero`, `acl`, ...

# 2. Implementation Proposal

## 2.1 Extra requirements for implementation
Here are the requirements for proposal:
- a common component should handle all the verbose printing logic to
  respect the separation of concern principle. This will allow to
  simplify the modification of the verbose format, or add a
  functionality without modifying all components (e.g. let user
  specify an output file instead of `stdout`).
- we should provide meaningful and actionable messages, so that user
  can troubleshoot their applications without requiring the
  intervention of a oneDNN developer.
- rely on a message catalog, so that consistent messages are provided
  independently of the backend and/or implementation emitting
  it. Furthermore, this helps with localization if someone wants to
  contribute message translations.

## A common interface to log messages

This could be achieved by providing a set of macros that can be used
within implementations. We chose macros as it allows to avoid
boilerplate code required to pass common information explicitly as
parameters (e.g. filename, line number, `pd()->info()` for dispatch
info, ...). Inside the macro call, we can still rely on functions to
avoid the library size from blowing up.

Those macros should not take strings as input except for
implementation specific information for developers. For common
messages, we should use enum values as it has 2 main benefits:
- makes common messages map to the same string, which simplifies
  parsing the verbose output to find an information, as there is an
  exact match between printed string and the kind of message it
  carries.
- simplifies potential future localization efforts.


Here is the list of macros that were identified as potential
candidates.  First check functions targeted at displaying
errors. Those should return a value when some condition is false and
they should typically be used when we return from an user visible
function.
- `CHECK(cat, cond, ret_val, msg_enum)`, will check the condition
  `cond`, emit the message according to the category of messages
  `cat`, and return `ret_val`. This macro would replace the existing
  `CHECK` macro.
- `<THIRDPARTY>_CHECK(msg_cat, cond, ret_val, msg_enum)` would allow to report
  failed checks from calls to third party dependencies
  (e.g. `SYCL_CHECK`, `OCL_CHECK`, `ACL_CHECK`, ...)

Another set of macro to display other kind of user oriented information:
- `PROFILE(msg_cat, )` These are mainly for printing creation and
  execution time.
- `DISPATCH_CHECK(cond, msg_enum)` would print msg in the log, with
  proper formatting. This macro is to be used in init function of each
  primitive implementation. The granularity is implementation defined,
  but general messaging should be enough (e.g. if only trivial strides
  / dense tensors are supported for an implementation, no need to
  specify which input does not satisfy the condition).  If the
  condition is false, this macro returns `unimplemented`.  This would
  be a syntactic sugar macro for `CHECK(dispatch, cond,
  status::unimplemented, msg_enum)`
- `ADVICE(cond, msg_enum)` this is for cases where we do dispatch our
  best performing implementation, but we can provide advice to user
  that they can change something to make it faster. A typical example
  is to let user know that they use memory with sub-optimal alignment,
  or in the case of Matmul, they could use different leading
  dimensions to improve performance.

A set of macros that are prefixed with `D`, which would be enabled
only when the library is built with `ONEDNN_DEVEL_MODE=ON`. Those
should be compiled away when the library is built with
`ONEDNN_DEVEL_MODE=OFF` (default).
- `DINFO(level, get_string)` to print developer info. Here
  `get_string` should be a function/lambda with the signature
  `std::string (void)`. This should allow to avoid string creation
  overheads when `ONEDNN_DEVEL_MODE=OFF`.

Some macros to collect timings and profile internal computations on host/cpu.
- `DDECLTIMER(timer)`, useful to declare a timer as primitive
  member, and collect performance across multiple runs of the same primitive.
- `DTIMESTAMP(level, stamp)` useful to add host timestamp in a way that there
  is no overheads when the library is not in `ONEDNN_DEVEL_MODE=OFF`.
- `DCOLLECTTIME(level, timer, start_stamp, end_stamp)`. This will add
  `end_stamp - start_stamp` to the timer.
- `DPROFILE(level, get_string, timer)`. Prints the info relative to a
  given timer.  Ideally, this should be called in primitive destructor
  to avoid print overheads.

Similar profiling macros but for device time.
- `DDEVICEDECLTIMER(timer)`, useful to declare a timer as primitive
  member, and collect performance across multiple runs of the same primitive.
- `DDEVICETIMESTAMP(stamp, event)` useful to add a timestamp on the
  device. This should not cause synchronization with the device. The
  current definition takes an event as input, but if too complex to
  retrieve events, we can simplify in different ways (e.g. using
  stream and inserting dummy kernel to get fresh event).
- `DDEVICECOLLECTTIME(level, timer, start_stamp, end_stamp)`. This
  should be asyncronous as well. To wait on event and collect profile
  info, it will likely need to spawn a new thread for sycl, and use
  callbacks for OCL.  It will also require some care to avoid race
  conditions (e.g. using std::atomic for some members, at least for a
  counter of collections currently waiting, and for the accumulated
  time).
- `DDEVICEPROFILE(level, get_string, timer)`. Same as `DPROFILE` but
  for reporting device time. This call will be a synchronous call, so
  it is recommended to use it only in primitive destructor to reduce
  overheads.


## Device side printing
Here the question is do we have any information to print from device
kernel code?

For nGEN and jit:ir implementation, there is no expectation that we
dump information during kernel execution.  For openCL, we might want
to dump tensors using printf. Unfortunately, openCL supports only
printf and no other function. So for now, we propose to let dumping in
OCL kernels be ad hoc, but should be guarded with the new
`ONEDNN_DEVEL_MODE` macro for uniformity.

# Summary

We propose to:
- introduce an internal API to handle verbose and message emitting in
  general. This should not impact oneDNN user. We will also introduce
  a new `ONEDNN_DEVEL_MODE` CMAKE option to simplify and align debug
  printing for developers.
- unify the messages format for errors. This is user visible, and
  might require to change scripts that parse errors (profiling
  information will not change format).
- change the `ONEDNN_VERBOSE` environment variable from levels to set
  of flags. We will maintain levels 0, 1 and 2 for smooth transition.
- add messages related to various errors, in particular for `op_desc`
  creation.
- add messages related to dispatching and why implementations don't
  dispatch. This will likely be coarse grain information at first, but
  we can work on refining it over time.
- add performance hints related to known factors impacting performance
  (memory alignment and leading dimensions in particular).
