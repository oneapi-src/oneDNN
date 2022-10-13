# Clamp {#dev_guide_op_clamp}

**Versioned name**: *Clamp-1*

**Category**: *Activation*

**Short description**: *Clamp* operation represents clipping activation
function.

**Detailed description**:

*Clamp* does the following with the input tensor element-wise:

  \f$ clamp( x_i )=min(max(x_i,min\_value),max\_value)\f$

## Attributes

* *min*

  * **Description**: *min* is the lower bound of values in the output. Any value
    in the input that is smaller than the bound, is replaced with the *min*
    value. For example, *min* equal 10 means that any value in the input that is
    smaller than the bound, is replaced by 10.
  * **Range of values**: arbitrary valid f32 value
  * **Type**: f32
  * **Required**: *yes*

* *max*

  * **Description**: *max* is the upper bound of values in the output. Any value
    in the input that is greater than the bound, is replaced with the *max*
    value. For example, *max* equals 50 means that any value in the input that
    is greater than the bound, is replaced by 50.
  * **Range of values**: arbitrary valid f32 value
  * **Type**: f32
  * **Required**: *yes*

## Inputs

* **1**: ``input`` - multidimensional input tensor. **Required.**

  * **Type**: T

## Outputs

* **1**: ``output`` - multidimensional output tensor with shape and type
  matching the input tensor. **Required.**

  * **Type**: T

**Types**:

* **T**: f32, f16, bf16.
* **Note**: Inputs and outputs have the same data type denoted by *T*. For
  example, if input is f32 tensor, then all other tensors have f32 data type.
