# RFC: f64 data_type support.

## Introduction & Motivation

Some deep learning applications require higher numerical accuracy than traditionally 
supported 32-bit floating points, especially for training phase. This RFC addresses
this requirement by introducing 64-bit, double precision float support in oneDNN.

### API changes

This is the list of the API changes required to add f64 data type support in oneDNN:
// put the code here.

## Some details on Convolution primitive

Convolution primitive in particular has several configurations and parameters that 
deserves to be discussed in more detail with regard to f64 data type.
...


### Additional changes

Maybe f64 data type support needs to be expanded for other primitives in oneDNN?
