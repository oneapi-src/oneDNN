# Proposal for Dilated Pooling in DNNL

## Introduction
Dilated pooling is simply regular pooling but the pixels/voxels you use in each
 "application" of the pooling operation are with defined gaps. The method of
 determining subsequent pixels is identical to the dilated
 convolution (implemented in DNNL).
![](dilated_pooling_example.png "Dilated Pooling Input Pixels (2x2 Pooling)")

### Motivation
Improve Pytorch performance, PyTorch Community argues Max_pool2d is slow in
native CPU path, so if MKLDNN can support it, may be can get a good performance.
- [MKL-DNN isuee](https://github.com/intel/mkl-dnn/issues/325)
- [PyTorch isuee](https://github.com/pytorch/pytorch/issues/34675)

### Simple measurement from the PyTorch community

> import torch  
> import time  
> import torch.nn as nn
> 
> model = 
nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, dilation=2)
> net = nn.MaxPool2d(kernel_size=3, stride=2, dilation=2)  
> blob = torch.randn(1, 16, 5000, 5000, device='cpu')
> 
> t0 = time.time()  
> with torch.no_grad():  
> outputs = model(blob)  
> print("PyTorch Conv: {}".format(time.time() - t0))
> 
> t0 = time.time()  
> with torch.no_grad():  
> pred = net(blob)  
> print("PyTorch MaxPool: {}".format(time.time() - t0))

> mkldnn = on
> with dilation = 2:
> conv2d 0.22s
> maxpool2d RuntimeError: std::exception
> without dilation:
> conv2d 0.21s
> maxpool2d 0.45s

> mkldnn = off
> with dilation = 2:
> conv2d 2.02s
> maxpool2d 2.32s
> without dilation:
> conv2d 0.35s
> maxpool2d 2.69s


## Overview

### Mathematical formula

#### Simple pooling (2d)
Max pooling:
```math
dst(n,c,oh,ow)=maxkh,kw(src(n,c,oh⋅SH+kh−PHL,ow⋅SW+kw−PWL))
```

Average pooling:
```math
dst(n,c,oh,ow)=1/DENOM∑kh,kwsrc(n,c,oh⋅SH+kh−PHL,ow⋅SW+kw−PWL)
```
#### Dilated pooling (2d)
Max pooling:
```math
dst(n,c,oh,ow)=maxkh,kw(src(n,c,oh⋅SH+kh * (DH + 1) −PHL,ow⋅SW+kw * (DW + 1) −PWL))
```

Average pooling:
```math
dst(n,c,oh,ow)=1/DENOM∑kh,kwsrc(n,c,oh⋅SH+kh * (DH + 1) −PHL,ow⋅SW+kw * (DW + 1) −PWL)
```
Where DH is dilation height and DW is dilation width, dilation starts with 0,
 so non-dilated convolution should have the dilation parameters equal 0.



### Support in key frameworks
|	TensorFlow	|	PyTorch	|	MxNet	|	PaddlePaddle	|
|--------------|--------------|--------------|--------------|
|tf.nn.pool|torch.nn.MaxPoolXd torch.nn.AvgPoolXd|Unsupported|Unsupported|

## Proposal
The proposal is modify the code of existing pooling to include dilation. 
We need to include dilation in the pixel-walking algorithm for 
all implementations. We can do this in our API in two ways.
### Option 1 (selected option)
Adding new parameter to function pooling_desc_init - dilation 
(solution as in the convolution) and create Add a new primitive 
(pooling_v2) with new descriptor that supports dilation. 
We maintain backward compatibility.
### Option 2
Find a creative way to pack dilation into e.g. strides or padding 
(they are 64-bit integers and strides, dilation and 
padding values are usually small).
### Option 3
(Maybe an option) Don't implement the primitive.
### Option 4
(Probably a non-option) Break the ABI.
### Selected solution
The solution is create new descriptor and call of pooling version 2. 
Primitive descriptor remains the same and calls the same implementations 
as old pooling. We add a condition that will allow implementation to be 
called from version 2 if dilated support is implemented.
### Work estimation
The main work to introduce this functionality will be choosing a solution, 
modifying the currently existing code and writing additional tests. We will 
also need to contact the communities to support this feature.
### High level execution steps
Depending on the implementation of the first or second option, you must 
either add the use of new functions or modify the existing implementation in 
libraries. The second solution seems definitely worse because of the need for 
changes in all libraries, in the first option it will only be needed in 
PyTorch and TensorFlow. In addition, it will not be necessary to change 
the operation of the entire pooling (option 2).