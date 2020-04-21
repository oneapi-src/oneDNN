# Inner Product Driver

## Usage
``` sh
    ./benchdnn --ip [benchdnn-knobs] [ip-knobs] [ip-desc] ...
```

where *ip-knobs* are:

 - `--dir={FWD_B [default], FWD_D, FWD_I, BWD_D, BWD_W, BWD_WB}`
            -- dnnl_prop_kind_t. Refer to the common glossary in README.md for
            details.
 - `--cfg={f32 [default], ...}` -- refer to ``Configurations`` in
            driver_conv.md.
 - `--stag={any [default], ...}` -- physical src memory layout.
            Refer to the common glossary in README.md for details.
 - `--wtag={any [default], ...}` -- physical wei memory layout.
            Refer to the common glossary in README.md for details.
 - `--dtag={any [default], ...}` -- physical dst memory layout.
            Refer to the common glossary in README.md for details.
 - `--attr="attr_str"` -- primitive attributes, default `""` (no attributes).
            Refer to knobs_attr.md for details.
 - `--mb=INT` -- override minibatch size specified in the problem description.
             When set to `0`, use minibatch size as defined by the individual
             problem descriptor. The default is `0`.

and *ip-desc* is a problem descriptor. The canonical form is:
```
    mbXicXidXihXiwXocXnS
```
Here `X` is an integer number and `S` is a string literal without spaces (`n`
stands for name). The special symbol `_` is ignored, so it may be used as a
delimiter for better readability. Refer to the common glossary in README.md for
the entity name and description.

There are default values for some entities in case they were not specified:
 - mb = 2;


## Essence of Testing
TBA.


## Examples

Run the set of ip from inputs/ip/ip_all file with default settings:
``` sh
    ./benchdnn --ip --batch=inputs/ip/ip_all
```

Run a named problem with single precision src and dst, backward by data
prop_kind, applying output scale of `2.25`, appending the result into dst with
output scale of `0.5`, and applying tanh as a post op:
``` sh
    ./benchdnn --ip --dir=BWD_D \
               --attr=oscale=common:2.25;post_ops='sum:0.5;tanh' \
               mb112ic2048_ih1iw1_oc1000_n"resnet:ip1"
```

More examples with different driver options can be found at
inputs/ip/test_ip_all. Examples with different driver descriptors can be found
at inputs/ip/ip_***. Examples with different benchdnn options can be found at
driver_conv.md.
