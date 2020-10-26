==========
Data Model
==========

oneDNN Graph uses logical tensor to describe data type, shape, and layout. The
data type could be fp32, int8, bf16, fp16, and future extension. The shape
contains multiple dimensions, and the total dimension and the size of dimension
could be set as unknown.

oneDNN Graph supports both public layout and opaque layout. When the layout_type
of logical tensor is “STRIDED”, it means that the tensor layout is public which
the user can identify each tensor element in the physical memory. For example,
for dims[][][] = {x, y, z}, strides[][][] = {s0, s1, s2}, the physical memory
location should be in s0*x+s1*y+s2*z.

When the layout_type of logical tensor is “OPAQUE”, users are not supposed to
interpret the memory buffer directly. A “OPAQUE” tensor can only be output from
oneDNN Graph’s compiled partition and can only be fed to another compile
partition as input tensor.
