.. SPDX-FileCopyrightText: 2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

---
End
---

**Versioned name**: *End-1*

**Category**: *Misc*

**Short description**: *End* operation doesn't have semantic and is used to track the uses of a tensor.

**Detailed description** 

For example, if an output tensor of an operation is not only used by another operation in the same graph,
but also used as an output of this graph, users can use **End** operation to handle multiple uses of this
output tensor. This is just one of typical use cases of this operation.


**Inputs**:

* **1**: input tensor. **Required.**
