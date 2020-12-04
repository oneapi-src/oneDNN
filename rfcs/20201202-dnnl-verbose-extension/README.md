# RFC: DNNL_VERBOSE functionality extension

## Motivation

oneDNN users wish to profile the library calls via chrome trace, which is
popular among different products. In order to have it working properly,
DNNL_VERBOSE should dump a timestamp of each call, otherwise it's impossible
to pin a call to a certain point in the timeline. It may be either an absolute
time value in milliseconds or relative to a certain start point dumped once.
Both ways will let to profile each call through the whole program execution.

## Proposal

To keep the current behavior intact, the proposal is to introduce a modifier
which will be enabled by additional environment variable
`DNNL_VERBOSE_TIMESTAMP`. When set to `0` (the default), it does not affect the
existing output of DNNL_VERBOSE. When set to `1`, it adds the timestamp field
into each call to primitive execute output.

Note:
The timestamp format varies based on implementation: on Linux the implementation
uses `gettimeofday` and returns time since the Unix epoch, on Windows it uses
`QueryPerformanceCounter` and returns time since the last system start.

There are two options where to put this timestamp:
* In front of the line. This follows the standard logging practices but will
  break existing parsing solutions. E.g.
  > 1607054150784.885010:dnnl_verbose,exec,cpu,reorder,rnn_data_reorder,undef,src_f32::blocked:abcd:f0 dst_u8::blocked:abcd:f0,,,2x3x224x224,3.3728
* As a second entry in the line. It complies the format of verbose line and
  looks nicer than the other two options (preferred). E.g.
  > dnnl_verbose,1607054150784.885010,exec,cpu,reorder,rnn_data_reorder,undef,src_f32::blocked:abcd:f0 dst_u8::blocked:abcd:f0,,,2x3x224x224,3.3728
* In the end of the line. This will preserve the existing solutions to work
  whether the variable is enabled or not, but may bring some confusion which
  time is actual performance, but which is a timestamp. Besides, to utilize the
  new ability, adjustments to script should be done any way. E.g.
  > dnnl_verbose,exec,cpu,reorder,rnn_data_reorder,undef,src_f32::blocked:abcd:f0 dst_u8::blocked:abcd:f0,,,2x3x224x224,3.3728,1607054150784.885010

## Additional information

This proposal moves the library to the option where each new modification should
be enabled by new environment variable, each properly documented alongside
`DNNL_VERBOSE` variable. The opposite approach may be using a single variable,
e.g. `DNNL_VERBOSE_MODIFIERS`, and union desired modifiers with a certain
delimiter, i.e. `|`. It makes parsing bit more difficult but potentially may
reduce coding the options interactions with each other. Though on user side it
wouldn't matter much which option we choose.

EOD.
