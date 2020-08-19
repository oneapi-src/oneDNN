====
Code
====

`LLGA on gitlab <https://gitlab.devtools.intel.com/pytorch-ats/llga>`__

Both the Spec and the code is still under development, so it may not reflect what we put in the spec. A few note below:

* LLGA:OP is implemented as LLGA:Node
* How the LLGA integration code modifies the Framework Graph to replace the partition related nodes with one custom Framework node is different than the pseudo code above. The custom Node inputs order may be different with LLGA Partition, the LLGA integration code is responsible to pass the input to LLGA execution API in the correct order(push to the LLGA execution stack with the order in Partition), since it can decide how to modify the graph and link the input to the custom Framework Node.
* The LLGA Partition defines its inputs and outputs, and contains a list of LLGA nodes. The mapping between Partition's inputs/outputs and each node's inputs/outputs are recorded in the LLGA values, e.g., if a Node's 1st input is the 6th input of the Partition, the Node and Partition will use same Value object(same id) for this input. So that the LLGA backend code can use this mapping info to get the offset of the input in the stack.

