# Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN)
![v0.95 beta](https://img.shields.io/badge/v0.90-beta-orange.svg)

> **Note**
>
> The master branch is now used to work on the upcoming Intel MKL-DNN v1.0 release with
> changes that are incompatible with v0.x. The changes are described in the following
> [RFC](https://github.com/intel/mkl-dnn/pull/384).
>
> For a limited time, the team will maintain
> [0.x branch](https://github.com/intel/mkl-dnn/tree/mnt-v0),
> backporting fixes and some of the features from the mainline.

Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN) is an
open-source performance library for deep-learning applications. The library
accelerates deep-learning applications and frameworks on Intel(R) architecture
and Intel(R) Processor Graphics Architecture. Intel MKL-DNN contains
vectorized and threaded building blocks that you can use to implement deep
neural networks (DNN) with C and C++ interfaces.

> **Note**
> Intel(R) MKL-DNN is distinct from Intel(R) MKL, which is general math performance
> library.

This release contains performance-critical functions that improve performance of the
following deep learning topologies and variations of these:

| Application                               | Example topology
|:---                                       |:---
| Image recognition                         | AlexNet, VGG, GoogleNet, ResNet, MobileNet
| Image segmentation                        | FCN, SegNet, MaskRCNN, U-Net
| Volumetric segmentation                   | 3D-Unet
| Object detection                          | SSD, Faster R-CNN, Yolo
| Neural machine translation                | GNMT
| Speech recognition                        | DeepSpeech
| Adversarial networks                      | DCGAN, 3DGAN
| Reinforcement learning                    | A3C
| Text-to-speech                            | WaveNet

Intel MKL-DNN is used in the following software products (please let us know if you are
using the library inside your appication so we can add to the list):
* [Caffe\* Optimized for Intel Architecture](https://github.com/intel/caffe)
* [Chainer\*](https://chainer.org)
* [DeepBench](https://github.com/baidu-research/DeepBench)
* [PaddlePaddle\*](http://www.paddlepaddle.org)
* [PyTorch\*](https://pytorch.org/)
* [Tensorflow\*](https://www.tensorflow.org)
* [Microsoft\* Cognitive Toolkit (CNTK)](https://docs.microsoft.com/en-us/cognitive-toolkit)
* [Apache\* MXNet](https://mxnet.apache.org)
* [OpenVINO(TM) toolkit](https://01.org/openvinotoolkit)
* [Intel Nervana Graph](https://github.com/NervanaSystems/ngraph)
* [Menoh\*](https://github.com/pfnet-research/menoh)
* [DeepLearning4J\*](https://deeplearning4j.org)
* [BigDL](https://github.com/intel-analytics/BigDL)

## License
Intel MKL-DNN is licensed under
[Apache License Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). This
software includes the following third-party components:
* [Xbyak](https://github.com/herumi/xbyak) distributed under
[3-clause BSD licence](src/cpu/xbyak/COPYRIGHT)
* [gtest](https://github.com/google/googletest) distributed under
[3-clause BSD license](tests/gtests/gtest/LICENSE)
* [ittnotify](https://github.com/intel/IntelSEAPI) distributed under
[3-clause BSD license](src/cpu/jit_utils/jitprofiling/LICENSE.BSD)

## Documentation
* [Developer guide](https://intel.github.io/mkl-dnn) explains programming
model, supported functionality, details of primitives implementations and
includes annotated examples.
* [API reference](https://intel.github.io/mkl-dnn/modules.html) provides
comprehensive reference of the library API.

## Support
Please submit your questions, feature requests, and bug reports on the
[GitHub issues](https://github.com/intel/mkl-dnn/issues) page.

> **WARNING**
> The following functionality has preview status and might change without prior
> notification in future releases.

* Threading Building Blocks (TBB) support
* Intel(R) Processor Graphics support

## How to Contribute
We welcome community contributions to Intel MKL-DNN. If you have an idea on how to improve
the library:

* Share your proposal via
 [GitHub issues](https://github.com/intel/mkl-dnn/issues).
* Ensure that you can build the product and run all the examples with your patch.
* In the case of a larger feature, create a test.
* Submit a [pull request](https://github.com/intel/mkl-dnn/pulls).

We will review your contribution and, if any additional fixes or modifications
are necessary, may provide feedback to guide you. When accepted, your pull
request will be merged to the repository.

## System Requirements
Intel MKL-DNN supports Intel 64 architecture and compatible architectures.
The library is optimized for the systems based on
* Intel Atom(R) processor with Intel SSE4.1 support
* 4th, 5th, 6th, 7th, and 8th generation Intel(R) Core(TM) processor
* Intel(R) Xeon(R) processor E3, E5, and E7 family (formerly Sandy Bridge,
Ivy Bridge, Haswell, and Broadwell)
* Intel(R) Xeon(R) Scalable processors (formerly Skylake and Cascade Lake)
* Intel(R) Xeon Phi(TM) processors (formerly Knights Landing and Knights Mill)

and compatible processors.

Intel MKL-DNN supports Intel(R) Processor Graphics.
The library is optimized for the systems based on
* Intel(R) Iris(R) Pro Graphics.

The software dependencies are:
* [CMake](https://cmake.org/download/) 2.8.11 or later
* [Doxygen](http://www.doxygen.nl/download.html#srcbin) 1.8.5 or later
* C++ compiler with C++11 standard support
* Optional dependencies:
  * GNU\* OpenMP\*, LLVM OpenMP, or Intel OpenMP
  * Threading Building Blocks (TBB) 2017 or later
  * Intel MKL 2017 Update 1 or Intel MKL small libraries

The additional software dependencies for Intel(R) Processor Graphics support:
* OpenCL\* runtime library (OpenCL\* version 1.2 or later)
* OpenCL\* driver (with kernel language support for OpenCL\* C 2.0 or later)
  with Intel(R) subgroups extension support

> **Note**
> Building Intel MKL-DNN with optional dependencies may introduce additional
> runtime dependencies for the library. For details, refer to the corresponding
> software system requirements.

The software was validated on RedHat\* Enterprise Linux 7 with
* GNU Compiler Collection 4.8, 5.4, 6.1, 7.2, and 8.1
* Clang\* 3.8.0
* [Intel C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  17.0, 18.0, and 19.0

on Windows Server\* 2012 R2 with
* Microsoft Visual C++ 14.0 (Visual Studio 2015 Update 3)
* [Intel C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  17.0 and 19.0

on macOS\* 10.13 (High Sierra) with
* Apple LLVM version 9.2 (XCode 9.2)
* [Intel C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  18.0 and 19.0

Intel(R) Processor Graphics support was validated on Ubuntu\* 18.04 with
* GNU Compiler Collection 5.4 and 8.1
* Clang\* 3.8.1
* [Intel C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  19.0
* [Intel(R) SDK for OpenCL\* applications](https://software.intel.com/en-us/intel-opencl) 2019 Update 3
* [Intel(R) Graphics Compute Runtime for OpenCL\*](https://github.com/intel/compute-runtime/releases) 19.15.12831

on Windows Server\* 2019 with
* Microsoft Visual C++ 14.0 (Visual Studio 2015 Update 3)
* [Intel C/C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe)
  19.0
* [Intel(R) SDK for OpenCL\* applications](https://software.intel.com/en-us/intel-opencl) 2019 Update 3
* [Intel(R) Graphics - Windows\* 10 DCH Drivers](https://downloadcenter.intel.com/download/28783/Intel-Graphics-Windows-10-DCH-Drivers) 26.20.100.6709

The implementation uses OpenMP 4.0 SIMD extensions. We recommend using the
Intel C++ Compiler for the best performance results.

--------

[Legal Information](doc/legal_information.md)
