==================
LLGA OP definition
==================

.. glossary::
   Conv2d

     The convolution operator consumes an input tensor and a filter, and computes the output.

     **Inputs**
     1 input : tensor(float16), tensor(float), tensor(double)
     Input data tensor from previous OP; has size (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and width. Note that this is for the 2D image.
     
     2 weight : tensor(float16), tensor(float), tensor(double)
     The weight tensor that will be used in the convolutions; has size (M x C/group x kH x kW), where C is the number of channels, and kH and kW are the height and width of the kernel, and M is the number of feature maps.
     
     3 bias(optional) : tensor(float16), tensor(float), tensor(double)
     Optional 1D bias to be added to the convolution, has size of M.
     
     **Attributes**
     1 stride : list of constant ints
     Stride along each spatial axis.
     
     2 padding : list of constant ints
     Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represents the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`.
     
     3 dilation : list of constant ints
     dilation value along each spatial axis of the filter.
     
     4 groups : list of constant ints
     number of groups input channels and output channels are divided into.
     
     5 kernel_shape : list of constant ints
     The shape of the convolution kernel. If not present, should be inferred from input W.
     
     **Output**
     Output : tensor(float16), tensor(float), tensor(double)
     Output data tensor that contains the result of the convolution.

     A LLGA OP with kind kconv2d represents a 2D convolution, which normally has 3 inputs Values as listed above (the kernel shape can be inferred) and 1 output Value. These values represent edges in DNN graph.

.. glossary::
   Relu
     Relu takes one input Tensor and produces one output Tensor where the rectified linear function, y = max(0, x), is applied to the tensor elementwise.

     **Inputs**
     input : tensor(float16), tensor(float), tensor(double)
     Input data tensor.

     **Output**
     Output : tensor(float16), tensor(float), tensor(double)
     Output data tensor that contains the result of the relu.

.. glossary::
   Pooling
     Pooling consumes an input tensor and applies max or average pooling across the tensor according to kernel sizes, stride sizes, and pad lengths, which means computing the max or average on all values of a subset of the input tensor according to the kernel size and downsampling the data into the output tensor Y for further processing.

     **Inputs**
     1 input : tensor(float16), tensor(float), tensor(double)
     Input data tensor from the previous OP.

     **Attributes**

     1 kernel_shape(attribute) : list of constant ints
     The size of the kernel .

     2 stride : list of constant ints
     Stride along each spatial axis.

     3 padding : list of constant ints
     Padding for the beginning and ending along each spatial axis, it can take any value greater than or equal to 0. The value represents the number of pixels added to the beginning and end part of the corresponding axis. `pads` format should be as follow [x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and xi_end, the number of pixels added at the end of axis `i`.

     4 ceil_mode : bool
     Whether to use ceil or floor (default) to compute the output shape.

     5 Algorithm : enum(MAX, AVG_INCLUDE_PADDING, AVG_EXCLUDE_PADDING)
     Specify the pooling mechanism.

     **Output**
     Output : tensor(float16), tensor(float), tensor(double)
     Output data tensor that contains the result of the pooling.

.. glossary::
   BatchNormalization
     Carries out batch normalization as described in the paper "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" at
     https://arxiv.org/abs/1502.03167.

     **Inputs**
     1 input : tensor(float16), tensor(float), tensor(double)

     2 weight(optional) : tensor(float16), tensor(float), tensor(double)
     The scale (γ).

     3 bias(optional) : tensor(float16), tensor(float), tensor(double)
     The shift (β).

     4 mean : tensor(float16), tensor(float), tensor(double)
     Running (training) or estimated (testing) mean tensor

     5 variance : tensor(float16), tensor(float), tensor(double)
     Running (training) or estimated (testing) variance tensor

     6 train: bool
     If set to true, run spatial batch normalization in training mode

     **Attributes**
     1 epsilon : float
     a constant to improve numerical stability.

     2 momentum : float
     Factor used in computing the running mean and variance.

     **Output**
     Output : tensor(float16), tensor(float), tensor(double)
     The output tensor of the same shape as input.

.. glossary::
   InnerProduct
     The inner product OP (sometimes called fully connected) computes input tensor’s product with a weights 2D tensor.

     **Inputs**
     1 input : tensor(float16), tensor(float), tensor(double)
     Input data tensor.

     2 weight : tensor(float16), tensor(float), tensor(double)
     The weight tensor that will be used in the product.

     3 bias(optional) : tensor(float16), tensor(float), tensor(double)
     Optional 1D bias to be added to the product.

     **Output**
     Output : tensor(float16), tensor(float), tensor(double)
     Output data tensor that contains the result of the InnerProduct.

.. glossary::
   Softmax
     This OP computes the softmax (normalized exponential) values for the given input along a particular dimension.

     **Inputs**
     1 input : tensor(float16), tensor(float), tensor(double)
     Input data tensor.

     2 dim : int
     The dimension along which softmax will be computed.

     **Output**
     Output : tensor(float16), tensor(float), tensor(double)
     Output data tensor that contains the result of the Softmax.

