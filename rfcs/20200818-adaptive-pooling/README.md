## Introduction
Adaptive pooling is simply pooling (max or average) where kernel size, strides and paddings are automatically adaptive to size of output image. In order to adaptive kernel size,
strides and paddings we are using two indexes, namely:
start_index=floor(a*input_size/output_size),
end_index=ceil((a+1)*input_size/output_size)
where a is an integer starting from 0 and less than outputsize.

Example of adaptive average pooling:
The input is the 1d image (1,2,3,4,5,6,7,8) of size 8. We want to use 1d adaptive average pooling to obtain 1d image of size 3. The adaptive average pooling 1d uses average
pooling 1d with padding and strides egual to one and kernel size equal to 4. In this case we have three following kernels, namely (0,1,2,3), (3,4,5,6), (6,7,8,0). Finally we obtain
the output (2,4.5,7)

## Motivation
DNNL doesn't support adaptive pooling. Adaptive average polling 2d is used in ResNet101 model.


## Overview

### Mathematical formula


#### simple pooling (2d)

n -batch
c - channels
oh - height
ow - width
kh - kernel height
kw - kernel width
SH - stride hight
SW - stride width
KW - width of kernel
KH - hight of kernel
PH_L - padding top
PW_L - padding left

Max pooling:
```math
dst(n,c,oh,ow)=max_{kh,kw}(src(n,c,oh*SH+kh-PH_L, ow*SW+kw-PW_L))
```

Average pooling:
dst(n,c,oh,ow)=1/DENOM\sum_{kh,kw}src(n,c,oh*SH+kh-PH_L, ow*SW+kw-PW_L)

where DENOM is equal to:
KH*KW if algorithm dnnl_pooling_avg_include_padding is enabled
KH*KW minus number of zeros coming from padding if algorithm dnnl_pooling_avg_exclude_padding is enabled

#### Adaptive pooling (2d)

IH - hight of input
OH - hight of output
IW - width of input
OW - width of output
alg - algotithm

##### Adaptive formula

SH = floor((2*IH)/(OH) - floor(IH/OH)
SW = floor((2*IW)/(OW) - floor(IW/OW)
KH = ceil((2*IH)/(OH) - floor(IH/OH)
KW = ceil((2*IW)/(OW) - floor(IW/OW)
PH_L = (SH*(OH - 1) + KH - IH)/2
PH_L = (SW*(OW - 1) + KW - IW)/2


Adaptive average pooling:
We applicate the adaptive formula to avegare pooling with alg equal to dnnl_pooling_avg_exclude_padding.

Adaptive max pooling:
We applicate the adaptive formula to avegare pooling with alg equal to dnnl_pooling_avg_exclude_padding.

	


## Proposal

The proposal not include any changes in DNNL code. The adaptive formula was added by Ma Jing on the client side in PyTorch.