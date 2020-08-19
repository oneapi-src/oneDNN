
Operation Set Specification
=================================
This specification document describes operation set supported.

+---------------+-------------------------------------------------+----------------------------------------------------------+ 
| Category      | oneDNN Graph                                    | Potential Candidates from Openvino                       |                                             
+===============+=================================================+==========================================================+
| activation    | BoundedRelu Elu Exp Gelu LogSoftmax Relu        | Clamp Elu HardSigmoid PReLU                              | 
|               | Sigmoid Softmax SoftRelu Swish Tanh             |                                                          |
+---------------+-------------------------------------------------+----------------------------------------------------------+
| arithmetic    | Abs Clip Linear Log Power Sqrt Square           | Acos Acosh Asin Asinh Atan Atanh Ceiling Cos Cosh Cumsum |
|               | Add Multiply Maximum Minimum                    | Divide Erf FloorMod Floor Mod Negative Sign Sin Sinh     | 
|               |                                                 | SquaredDifference Subtract                               |
+---------------+-------------------------------------------------+----------------------------------------------------------+
| comparison    |                                                 | Equal GreaterEqual Greater LessEqual Less NotEqual       |
+---------------+-------------------------------------------------+----------------------------------------------------------+
| condition     |                                                 | Bucketize NonZero Select                                 |
+---------------+-------------------------------------------------+----------------------------------------------------------+
| convolution   | Convolution Deconvolution GroupConvolution      | BinaryConvolution DeformableConvolution                  |
+---------------+-------------------------------------------------+----------------------------------------------------------+
| detection     |                                                 | DeformablePSROIPooling DetectionOutput PSROIPooling      |
|               |                                                 | PriorBoxClustered PriorBox Proposal ROIAlign ROIPooling  |
|               |                                                 | RegionYolo ReorgYolo                                     |
+---------------+-------------------------------------------------+----------------------------------------------------------+
| generation    |                                                 | Range                                                    |
+---------------+-------------------------------------------------+----------------------------------------------------------+
| infrastructure|                                                 | Assign Constant Parameter ReadValue Result TensorIterator|
+---------------+-------------------------------------------------+----------------------------------------------------------+
| image         | Interpolate                                     |                                                          |
+---------------+-------------------------------------------------+----------------------------------------------------------+
| logical       |                                                 | LogicalAnd LogicalNot LogicalOr LogicalXor               |
+---------------+-------------------------------------------------+----------------------------------------------------------+
| matrix        | MatMul Sum                                      |                                                          |
+---------------+-------------------------------------------------+----------------------------------------------------------+
| misc          | WildcardOp CustomOp Contraction InnerProduct    |                                                          |
+---------------+-------------------------------------------------+----------------------------------------------------------+
| movement      | Concat ShuffleChannels                          | BatchToSpace Broadcast DepthToSpace GatherTree           |  
|               |                                                 | Gather Pad ReverseSequence Reverse ScatterElementsUpdate | 
|               |                                                 | ScatterNDUpdate ScatterUpdate SpaceToBatch SpaceToDepth  |
|               |                                                 | Split StridedSlice Tile Transpose VariadicSplit          |            
+---------------+-------------------------------------------------+----------------------------------------------------------+
| normalization | BatchNorm LayerNorm LRN                         | GRN MVN NormalizeL2                                      |
+---------------+-------------------------------------------------+----------------------------------------------------------+
| pooling       | AvgPool MaxPool                                 |                                                          |
+---------------+-------------------------------------------------+----------------------------------------------------------+
| quantization  |                                                 | FakeQuantize                                             |
+---------------+-------------------------------------------------+----------------------------------------------------------+
| reduction     |                                                 | ReduceLogicalAnd ReduceLogicalOr ReduceMax ReduceMean    | 
|               |                                                 | ReduceMin ReduceProd ReduceSum                           |
+---------------+-------------------------------------------------+----------------------------------------------------------+
| sequence      | GRUCell LSTMCell RNNCell LBRGruCell             | CTCGreedyDecoder LSTMSequence OneHot                     |
+---------------+-------------------------------------------------+----------------------------------------------------------+
| shape         |                                                 | Reshape ShapeOf Squeeze Unsqueeze                        |
+---------------+-------------------------------------------------+----------------------------------------------------------+
| sort          |                                                 | NonMaxSuppression TopK                                   |
+---------------+-------------------------------------------------+----------------------------------------------------------+
| sparse        |                                                 | EmbeddingBagOffsetsSum EmbeddingBagPackedSum             |
|               |                                                 | EmbeddingSegmentsSum                                     |
+---------------+-------------------------------------------------+----------------------------------------------------------+
| type          |                                                 | ConvertLike Convert                                      |
+---------------+-------------------------------------------------+----------------------------------------------------------+
