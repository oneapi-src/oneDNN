file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp
    )

set_source_files_properties(${SOURCES}
    PROPERTIES COMPILE_FLAGS "-O3 -funroll-loops")

set(OBJ_LIB ${DNNL_LIBRARY_NAME}_cpu_ppc64)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
