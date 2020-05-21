===================
Fall back mechanism
===================

The LLGA partition may not be able to compile or execute since the LLGA partition is formed with incomplete information. As the LLGA partition is determined at the graph optimization stage, the input tensor shape may or may not be available. When it is compiled with the input shape information, the compilation may fail since some input shape is not supported.  Even for a successfully compiled partition, it may fail at the execution time since the partition execution may generate intermediate tensor with dynamic tensor shape according to the input tensor value, like a different image may produce different number of region proposal in object detection model.

The LLGA integration layer calls LLGA partition compilation API and returns a failure, so the compiled LLGA partition could point to null object. When the LLGA integration layer code executes a null compiled partition, or when the LLGA integration layer calls LLGA partition execution API and returns a failure, it falls back to the framework. The LLGA integration layer code keeps track of the original framework subgraph. It falls back to the framework and execute the original framework subgraph encapsulated by the LLGA partition.

One possible risk is that the failed LLGA partition execution touched the input tensors, so that the fallback execution gets wrong inputs. Pytorch operation supports inplace operation which allows the operation to modify itself. When a partition is formed, the partition should not start with inplace operation.

