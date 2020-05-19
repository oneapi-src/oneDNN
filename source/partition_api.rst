=============
Partition API
=============

.. literalinclude:: code_snippets/partition_api.cpp
    :language: cpp

**Discussions:** A more advanced API might involve specifying the partition policy for different OP id through OP id lists. This supports users (framework developers)  to perform binary search to debug a potential issue in the compiled partition.

---------------------------------
Framework integration Pseudo Code
---------------------------------

.. code::

   All_partitions = {}
   for each node “s” in framework graph with topology order
     s’ = create_llga_op(s)
     If (s’ creation failed) || (s already visited) || (Select(s’) failed) then continue;
     G = new vector [] with initial (s,s’) OP pair
     while G changed
       for all unvisited OP “n” in G with DFS order
         for each input node “i” of n
           i’ = create_llga_op(i)
           if (select_input(i’, i_o_ind, n’, n_i_ind)) add (i, i’) to G
         for each output node “o” of n
           i’ = create_llga_op(i)
           if (select_output(o’, o_i_ind, n, n_o_ind)) add (o,o’) to G
     mark (n, n’) visited
     G = remove_cyclic_dependency(G)
     All_partitions = filter(G)
   for each partition P in All_partitions
     Create a new node P_NODE for P
     for each LLGA OP op’ inside P,
       If op’ input and output is in the P’s input/output list
   Find op from the OP pair (op, op’)
   Modify the framework graph to connect op’s input/output to the new node

**Discussion:** The select API doesn’t construct the LLGA graph, instead it passes the LLGA op and using the select input/output to help the LLGA backend to contract the graph. The select API fits a graph with dataflow relationship. When the graph becomes big and it may contain control flow graph, we will need to extend the select API to pass the control flow information. 

The other alternative is to have graph construction API which construct a LLGA graph and passes the graph to the backend. One likely scenario is that the LLGA adopt MLIR representation so that we can lower the framework MLIR to LLGA dialect, and then pass the MLIR with LLGA dialect to backend for partition. 

