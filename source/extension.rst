=========
Extension
=========

LLGA extension is to support devices which is not DPC++ compliant. Depending on the development stage of the SW stack, some devices may not support DPC++ language. The supported language may be enough to support focused customer but not positioned as a low-level programming model to enable the DL ecosystem. As the effort of bring a new device and language to framework is high, these devices may choose to integrate with framework as an ``implicit device``. In ``implicit device`` integration mode, framework is not aware of the existence of the device, and LLGA offloads a subgraph on CPU device to the acceleration HW.

1.  ``llga:send`` and ``llga:receive`` op to support efficient host/device memory transfer [TBD]

When the LLGA partition is supposed to offload to a different device, it needs to change the existing input and output protocol. Between the LLGA partition and framework CPU graph, the LLGA integration layer needs to insert send op in one side and receive op the other to facilitate the data transfer. Although logically it is the same tensor passing between send and receive op, but the tensor are in two devices, may be associated with multiple buffers, in different format.

``llga:send`` and ``llga:receive`` are introduced as LLGA extensions to facilitate the new protocol from the LLGA partition. One the framework CPU graph, the send and receive OP are replaced to ``llga:send`` and ``llga:receive`` ops.

The compiled LLGA partition needs to be extended to support a different tensor input output mapping.  The partition input memory may be in device, and the partition output memory may be in CPU. Inside the partition, it needs to add the receive node at the input boundary and send node at output boundary. The receive and send op may input and output memory many times larger than partition’s input/output tensor size. At the execution time, the framework prepares both input memory and output memory to bind to the partition parameters.

Since the ``llga:send``, ``llga:receive``, and LLGA partition are all implemented by LLGA backend, the the host device memory transfer protocol is not part of LLGA interface definition. It gives flexibility for LLGA backend to implement its own protocol. For example, how the device buffer address and size are decided, and possible lambda function which reorders the data format if device side expects a different format.

2. Extendable LLGA OP

There are cases where the LLGA backend may want to introduce new LLGA operations. For example, LLGA backend may want to show performance in a specific customer engagement or a domain. “Implicit device” may want to include more LLGA OP than “explicit device”, as none LLGA op on “implicit device” falls back to CPU and leads a significant slow down due to extra cost of host device data transfer.

For the OP beyond the LLGA OP, LLGA integration layer calls the extension to decide whether the framework op can be represented as an extended LLGA op.

3. Custom OP Registration

LLGA extension supports custom op description and so that the bridge can fill the information of framework custom op and pass to LLGA backend.  The LLGA custom op may have a variable number of input/output tensors and attributes and a string describing the logic using language like PlaidML Tile.

