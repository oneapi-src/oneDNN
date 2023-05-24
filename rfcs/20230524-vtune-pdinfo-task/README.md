Proposal for Vtune integration extension

# 1. Problem statement

The main way for oneDNN to efficiently collect information during an
application run is to use oneDNN verbose mode. This generates one line
per primitive execution, and allow for easily create reproducers using
benchdnn.


However, it is hard to correlate verbose output to the overall
application execution. Vtune in general does a better job at that as
it contains call stack and nested task information.

So currently, users are left to run Vtune for profiling their whole
application, and try to map primitive execution calls in the Vtune
trace to the oneDNN verbose output. This is tedious and error prone.

Furthermore, Vtune integration is enabled by default whereas oneDNN
verbose is not. So in practice, users collect Vtune information, ask
the oneDNN team to investigate an issue, and are left with no way to
generate a simple reproducer which stalls investigation and problem
resolution. At that point they have to rerun their application with
ONEDNN_VERBOSE enabled, which can be impossible sometime due to
limited availabily of hardware, or the duration of the workload.

# 2. Proposal

The proposal is to rely on the existing Vtune integration in oneDNN
and expose a new level `ONEDNN_ITT_TASK_LEVEL=3`, that would create a
task per unique primitive descriptor (`pd->info()` internally). This
way each task under Vtune would contain similar information as oneDNN
verbose output, which in turn would allow to easily obtain benchdnn
reproducers from Vtune trace.

Because levels are inclusive, oneDNN will use nested itt tasks, so
from Vtune trace, the user will get each primitives grouped by
primitive kind task, then within each of these groups they will get
the list of primitives effectively executed as shown in verbose trace
(so unique primitive descriptor).
