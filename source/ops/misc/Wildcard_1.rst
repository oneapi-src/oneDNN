.. SPDX-FileCopyrightText: 2020-2021 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

--------
Wildcard
--------

**Versioned name**: *Wildcard-1*

**Category**: *Misc*

**Short description**: *Wildcard* operation represents any compute logic and its
  input and output tensors contribute to the graph building.

**Attributes**

* *alias*

  * **Description**: A list of strings of comma-separated indices so that the
    first index is the input index and the remaining indices are output indices
    aliasing this input. Take `aten::relu_` as an example, the alias attribute
    would be: alias: ["0,0"] Here "0,0" means that input 0 (the first "0") is
    aliased by output 0 (the second "0").
  * **Range of values**: strings of comma-separated indices
  * **Type**: string[]
  * **Default value**: None
  * **Required**: *no*

* *mutation*

  * **Description**: A list of input tensor indices that might be mutated by the
    Op. Take `aten::relu_` as an example, the mutation attribute would be:
    Mutation: [0]
  * **Range of values**: Non-negative s64 value.
  * **Type**: s64[]
  * **Default value**: None
  * **Required**: *no*

* *op_name*

  * **Description**: indicate which OP Wildcard represents.
  * **Range of values**: string of op name
  * **Type**: string
  * **Default value**: None
  * **Required**: *no*

* *attr_names*

  * **Description**: list of attributes' name of represented OP
  * **Range of values**: strings of attributes' names
  * **Type**: string[]
  * **Default value**: None
  * **Required**: *no*

* *attr_types*

  * **Description**: list of attr_names' date type. The order of attr_types
    should match attr_names'. For example, the first value in attr_types list
    indicates the data type of the first attribute in attr_names list.
  * **Range of values**: strings of data types:"i","is","f","fs","s","ss","b",
    "bs", which means "int","int[]","float","float[]","string","string[]",
    "bool","bool[]"
  * **Type**: string[]
  * **Default value**: None
  * **Required**: *no*

**Inputs**:

* **0 - N**: input tensor. **Optional.**
  
  * **Type**: T

**Outputs**

* **0 - N**: output tensor. **Optional.**
  
  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.