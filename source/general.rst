============
Introduction
============

LLGA (Low level graph API) is a graph API as oneDNN extension to optimize deep learning (DL) applications. oneAPI is a single, unified programming model that aims to simplify development across multiple architectures – such as CPUs, GPUs, FPGAs and accelerators.  For the DL application development, oneAPI includes Data Parallel C++ language and specification for primitive API for deep learning compute and communication operations. Intel’s implementation of oneAPI includes DPC++ compiler and libraries, oneDNN, and oneCCL.

LLGA defines a standard API layer which separates target specific optimization from framework graph optimization. With LLGA API, the framework graph is lowered to an LLGA graph which is then passed to an LLGA backend. The optimization and code generation developed in LLGA backend is target specific and can be used for multiple DL frameworks as a separate and independent module. The LLGA API allows DL HW vendors to focus on the performance value by generating the best optimized binary for the target device. With a single API to support multiple HW classes, it also allows the framework to reduce the integration efforts and focus on common optimizations.

------------------------
Motivation - Performance
------------------------

DL applications are built on top of DL frameworks like tensorflow, pytorch, and a few others. Frameworks provide a rich set of deep neural network (DNN) operations so DL app developers can readily use to describe a DL model. They also allow developers to define and use custom DNN operations.  Most execution time of DL applications is spent on the DNN model, which is represented as a graph of DNN ops. LLGA is to accelerate the DNN model execution through optimizing DNN subgraph for DL HW backend.

The current oneAPI spec only contains primitive APIs like oneDNN and oneCCL. They are integrated to DL framework’s DNN operation execution functions and focus on optimizing the most frequently executed operations. For the compute vision workload, which is the first adopting DL techniques, it is dominated by a few very high compute intensive operations, primitive based API captures the key optimization problem and solves it very efficiently at the operation level. As most DNN models consist of alternating sequences of compute intensive op and memory insensitive op, the primitive APIs also provide fusion APIs to capture the most frequently executed operation pairs. However, the fusion API is specific to a few patterns and can’t capture general cross op optimization opportunities.

Graph level optimization is critical to exploit cross-op optimization opportunities and ensures good data locality for overall operations in the graph. With the rapid introduction of DL acceleration HW and new algorithms, the DL workload characteristic changed significantly. These combined SW and HW efforts lower down the compute intensive operation execution time, and which makes hot spots scattering among a broad set of operations. Accelerating a few compute intensive operations using primitive API has diminishing return and limits the performance potential.

The DL HW advancement drives the increasing needs for getting a graph of operations as input for maximum compute efficiency. As most DL HWs accelerate compute function units far more aggressively than increase cache capacity and memory BW, the hot spots shift from the GEMM based operation to memory bound operations. The hot spots also shift from a few large operations to a number of small operations. How well memory bound operations and small operations are executed has increasingly high impact to overall performance. Some more advanced DL HW contains vector, matrix, and DMA function units, and executes a sequence of DNN ops simultaneously and in a pipeline manner. For these DL HW, graph level software optimization is required to schedule the operations and orchestrate the execution, so the data is accessed in the local memory with highest possible bandwidth and best latency.

Besides the DL HW improvement, the DL SW algorithm also evolved significantly. With the wide adoption of deep learning techniques beyond compute vision, many new DL models were introduced with new DNN operations and patterns, and many new DL models and applications have lower compute intensity than the initial compute vision field. To achieve higher model accuracy with lower computation cost, some very high compute intensive ops are replaced with lower compute intensive ops. For example, Resnext uses Group Conv to replace regular Conv, and Transformer uses Batch Matmul instead of Matmul. For the new domain like recommendation and language processing, the data inputs and activations may be in low dimension, which can lead to low compute intensity especially for online low latency inference usage with low batch size. The shift of workload characteristic to low compute intensive operation requires more aggressive fusion.

----------------------
Separation of concerns
----------------------

LLGA focuses on graph level optimization specific to DL HW, also known as target specific or device specific. At this stage, the industry doesn’t have a clear winner of DL HW microarchitecture fitting the performance characteristic of all DL applications.  DL HWs have different memory subsystems and vector/matrix HW function units. On the software side, there is no common standard software module which can map a graph of DNN operations to different DL HW backends.  Each framework develops its internal representation (IR) for DL models and builds software on top of vendor provided primitive level performance libraries or low-level programming language.

Without LLGA, the default integration approach is to directly optimize framework graphs.  As a framework graph optimization module allows it to be extended to support target specific optimization on framework specific IR, it works fine when the target specific optimization is limited, like supporting a few popular fusion patterns. But when the target specific optimization becomes more complex, it involves major changes to the framework graph optimizer and IR representation, which takes a long time to get accepted. Having a well defined interface motivates the development of advanced target specific optimization from the DL HW vendor, and allows framework to get access to higher performance without significant amount of work.

------------
Adoptability
------------

LLGA doesn’t aim to be the runtime of target devices and become the only way framework can manage the device resources and use the device to compute. To be able to integrate a device framework, the framework needs to represent the device properly and is able to control the device resources like memory and command queue.  LLGA accepts the device runtime context specified by the framework, like device memory and command queue. The resource usage of LLGA execution is completely under the governance of the framework, and any extra resource allocation needs to be coordinated by framework runtime.

Low level graph API (LLGA) doesn’t mean that LLGA operation has lower level semantics than the framework operation. The low level property of LLGA is mainly defined by its focus on target dependent optimization and ability to work with device specified runtime context.

As part of oneAPI, LLGA is required to work together with DPC++ language and runtime. But LLGA API has an extension to work with a device which doesn’t have DPC++ support. LLGA extension API allows device without DPC++ support to be integrate with framework with same API, which ensures smooth transition from HW without DPC++ to HW with DPC++.

------
Values
------

The LLGA pathfinding is to prove the potential value listed below.

* LLGA will improve time to market for target specific optimizations through an properly defined interface an cross-op optimization opportunities
* Giving framework control, better performance, and fitting into framework runtime will lower the SW adoption barrier when HW gets adopted
* Single API from Intel to support multiple HW classes will lower maintenance for framework
