file(REMOVE_RECURSE
  "libdnnl.pdb"
  "libdnnl.so"
  "libdnnl.so.3"
  "libdnnl.so.3.5"
)

# Per-language clean rules from dependency scanning.
foreach(lang ASM C CXX)
  include(CMakeFiles/dnnl.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
