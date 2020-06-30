# Input files

**Benchdnn** supports input files to specify options and problems to test. File
names convention and naming rules are specified the following way:

* **shapes_\<label\>**: a file containing one or more input shapes specified as
either problem descriptor or dimensions, e.g. convolution shape
`ic16ih10oc32oh10kh3sh1ph1n"conv_1"`. Such files should contain only shapes and
cannot have any driver options specified, such as data types or directions.

* **set_\<label\>**: a file containing one or more **shapes_\<label\>** files or
other sets. Such files should contain only input shapes and cannot have any
driver options specified, such as data types or directions. The general rule is
to group single-feature inputs into a single "batch", i.e. all *topology* based
inputs, all *regression* based inputs, all 2D-spatial problems, etc.

* **option_set_\<label\>**: a file similar to **set_\<label\>**, but it should
contain other driver options. If no driver options are specified, it should be
converted into a **set_\<label\>** file. The general rule is to unite repeated
configurations into a single "batch", i.e. all eltwise algorithms with valid
alpha and beta values, etc.

* **harness_\<driver\>_\<label\>**: a deployable suite of configurations and
sets or shapes. Entries in a harness file may include many instances and
combinations of driver options and/or batch files, including other harnesses.
Harness *must* have the `--reset` option at the beginning of the file to avoid
option collision. The general idea is to unite multiple cases to test a single
but broad feature, e.g. attributes for a certain driver.

* **test_\<driver\>_\<label\>**: These files are used for deploying correctness
testing via command-line `make <test>`. Entries in a test file are preferred
to be harnesses but may contain the same content as harness files. Test files
*must* have the `--reset` option at the beginning of the file to avoid option
collision.

* **perf_\<driver\>_\<label\>**: These files are used for deploying performance
testing via command-line `make <perf>`. The content of the file is the same as
for test files. Perf files *must* have the `--reset` option at the beginning of
the file to avoid option collision.
