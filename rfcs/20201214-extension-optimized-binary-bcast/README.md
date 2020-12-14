# RFC: Extension of the optimized binary with broadcast

## Motivation

OneDNN users report the need to add optimized implementations for different
broadcast combinations in the binary. Currently, jit version supports only 
such cases: NxCxDxHxW:{NxCx1x1x1,1xCx1x1x1,1x1x1x1x1}. Therefore, the best 
option would be to find a technical solution to use any case of broadcast. But 
for the best optimization and avoiding the next limitations, it is also possible
to add further needed broadcasting methods in the jit kernel.

## Proposal

As mentioned there are two general approaches to the problem and different 
solutions:
1. Extension of the current available cases.
    - Pros:
        - Simpler solution.
        - Better optimization possible.
    - Cons:
        - Require new implementation for each subsequent need from the users.
        - Complication of next cases implementations.
        - Complication for the user who needs to know if this case is optimized.
    - Implementation - a dedicated extension for a given new case. There may be 
      a problem with broadcast channel in block format.
2. Allow any broadcast in jit version.
    - Pros:
        - Allowing all possible broadcast options.
        - Unify the optimized version.
    - Cons:
        - More difficult solution.
        - Less optimization for generic version.
        - Other limitations possible.
    - Implementation
        - Calculate the index of next elements to load or broadcast to vector. 
          Dividing work into threads according to no broadcasted dimensions.
        - Copy source1 to a destination memory with broadcast (to scratchpad if 
          in-place mode or disable in-place mode) and do binary operation 
          between source0 and destination for better vectorization.
