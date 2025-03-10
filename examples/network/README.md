# PointNet Convolutional Neural Network Sample for 3D Pointcloud Classification

[PointNet][pointnet-paper] is a convolutional neural network architecture for applications concerning 3D recognition such as object classification and part segmentation. These sample codes implement a variant of PointNet for 3D object classification, for inference only with ModelNet10, showing a larger example of using oneDNN. Some rough instructions for how it might be used are provided.

## Obtaining the model weights and classes and preparing an input pointcloud

A preprocessing script is provided which unpacks the weights from a pretrained pytorch model. The script also prepares an input pointcloud for testing inference. The pointcloud is made from 3D scans taken from the [ModelNet10][modelnet] dataset. The script requires an installation of [PyTorch][pytorch]. First download the pretrained PointNet weights and move the pth file into the same directory of the model.

```bash
wget http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
unzip ModelNet10.zip
rm ModelNet10.zip
python3 prepareData.py ModelNet10/ pointnet_model.pth
```

The weights will be saved to `data/` and the input pointcloud will be saved as `itemName_cloud.bin`

## Testing on a pointcloud

The oneDNN samples are built in the default CMake configuration. The sample
is built by the target `network-pointnet-cpp`. The samples must first
be passed the directory where the binary weights files are stored and the second
argument should be the preprocessed pointcloud that should be classified. The expected
output is of a classification index and a series of times in nanoseconds that corresond
to the total time to run the network on an input, not including data transfer time.

```bash
network-pointnet-cpp ModelNet10/directory/extracted_data ModelNet10/directory/input_cloud/itemName_cloud.bin

```

[pointnet-paper]: https://arxiv.org/pdf/1612.00593.pdf
[pytorch]: https://pytorch.org/
[modelnet]: https://modelnet.cs.princeton.edu/

