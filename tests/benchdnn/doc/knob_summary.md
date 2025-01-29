# Summary information

## Usage
```
    --summary=[no-]SETTING1[+[no-]SETTING2...]
```

The `--summary` knob is a global state of benchdnn and provides the summary
statistics at the end of the run. Different options are separated with `+`
delimiter. To negate the effect of the option, use the "no-" prefix in front of
the option value.

If the same setting is specified multiple times, only the latter value is
considered.

## Failed cases summary

### Introduction
A batch file can contain a large number of test cases. Running the batch file
may result in the data in the beginning of the list getting lost due to the
short session screen buffer. Even if the buffer fits all the test cases in it,
to find the specific failed or unimplemented cases in the middle of the output
usually requires additional steps such as copying the whole screen buffer to a
memory buffer, collecting the data in a file, and using the search functionality
to scan through the file.

The `failures` knob can improve usability in such scenarios.

### Usage
```
    --summary=[no-]failures
```

By default, you can see the summary with up to ten failed cases.

To disable the summary output, use the "no-failures" input value.
