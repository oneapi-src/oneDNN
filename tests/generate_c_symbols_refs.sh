#!/bin/sh

mkl_dnn_root="$1"
output="$2"

echo -e '#include "mkl_dnn.h"' > "$output"
echo -e "const void *c_functions[] = {" >> "$output"
cpp "${mkl_dnn_root}/api/mkl_dnn.h" | /usr/bin/grep -o 'mkl_dnn_\w\+(' \
    | sed 's/\(.*\)(/(void*)\1,/g' | sort -u >> "$output"
echo -e "NULL};\nint main() { return 0; }" >> "$output"
