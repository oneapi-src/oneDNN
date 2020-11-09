==========
Data Model
==========

oneDNN Graph uses logical tensor to describe data type, shape, and layout. The
data type could be FP32, INT8, BF16, FP16, and future extension. The shape
contains multiple dimensions, and the total dimension and the size of the
dimension could be set as unknown.

oneDNN Graph supports both public layout and opaque layout. When the
``layout_type`` of logical tensor is ``strided``, it means that the tensor
layout is public which the user can identify each tensor element in the physical
memory.

For example, for :math:`dims[][][] = {x, y, z}`,
:math:`strides[][][] = {s0, s1, s2}`, the physical memory location should be in
:math:`s0*x+s1*y+s2*z`.

When the ``layout_type`` of logical tensor is ``opaque``, users are not supposed
to interpret the memory buffer directly. An ``opaque`` tensor can only be output
from oneDNN Graph’s compiled partition and can only be fed to another compile
partition as an input tensor.
