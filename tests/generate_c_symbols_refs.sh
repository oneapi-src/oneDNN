#!/bin/sh

mkldnn_root="$1"
output="$2"

echo -e '#include "mkldnn.h"' > "$output"
echo -e "const void *c_functions[] = {" >> "$output"
cpp "${mkldnn_root}/api/mkldnn.h" | /usr/bin/grep -o 'mkldnn_\w\+(' \
    | sed 's/\(.*\)(/(void*)\1,/g' | sort -u >> "$output"
echo -e "NULL};\nint main() { return 0; }" >> "$output"
