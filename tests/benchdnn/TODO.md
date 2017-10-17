# benchdnn Todo

* common:
    - add verbosity control through environment variable

* correctness:
    - change int to double for cfg->{min, max}
    - add quick testing
    - avoid segfault from errors in input/output height/width setting

* performance:
    - add efficiency output

* documentation:
    - add more examples on convolution notation

# Done

* fix float overflow

* add skip_impl option

* add input files

* add support for conv_relu

* add support for winograd

* add dilated convolution

* add `_` as delimiter for conv description (can we read it now?)

* add performance testing

* add 'all' flag to bench\_mode to compare all available impls
  - and a 'test' flag for alternate reference loop impls
  - add short name of convolution impl to performance output
