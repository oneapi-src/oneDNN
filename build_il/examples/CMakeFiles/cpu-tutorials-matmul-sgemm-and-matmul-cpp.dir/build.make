# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/shreyas/G/shr-fuj/oneDNN_open_source

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il

# Include any dependencies generated for this target.
include examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/progress.make

# Include the compile flags for this target's objects.
include examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/flags.make

examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/tutorials/matmul/cpu_sgemm_and_matmul.cpp.o: examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/flags.make
examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/tutorials/matmul/cpu_sgemm_and_matmul.cpp.o: ../examples/tutorials/matmul/cpu_sgemm_and_matmul.cpp
examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/tutorials/matmul/cpu_sgemm_and_matmul.cpp.o: examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/tutorials/matmul/cpu_sgemm_and_matmul.cpp.o"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/tutorials/matmul/cpu_sgemm_and_matmul.cpp.o -MF CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/tutorials/matmul/cpu_sgemm_and_matmul.cpp.o.d -o CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/tutorials/matmul/cpu_sgemm_and_matmul.cpp.o -c /home/shreyas/G/shr-fuj/oneDNN_open_source/examples/tutorials/matmul/cpu_sgemm_and_matmul.cpp

examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/tutorials/matmul/cpu_sgemm_and_matmul.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/tutorials/matmul/cpu_sgemm_and_matmul.cpp.i"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shreyas/G/shr-fuj/oneDNN_open_source/examples/tutorials/matmul/cpu_sgemm_and_matmul.cpp > CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/tutorials/matmul/cpu_sgemm_and_matmul.cpp.i

examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/tutorials/matmul/cpu_sgemm_and_matmul.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/tutorials/matmul/cpu_sgemm_and_matmul.cpp.s"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shreyas/G/shr-fuj/oneDNN_open_source/examples/tutorials/matmul/cpu_sgemm_and_matmul.cpp -o CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/tutorials/matmul/cpu_sgemm_and_matmul.cpp.s

# Object files for target cpu-tutorials-matmul-sgemm-and-matmul-cpp
cpu__tutorials__matmul__sgemm__and__matmul__cpp_OBJECTS = \
"CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/tutorials/matmul/cpu_sgemm_and_matmul.cpp.o"

# External object files for target cpu-tutorials-matmul-sgemm-and-matmul-cpp
cpu__tutorials__matmul__sgemm__and__matmul__cpp_EXTERNAL_OBJECTS =

examples/cpu-tutorials-matmul-sgemm-and-matmul-cpp: examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/tutorials/matmul/cpu_sgemm_and_matmul.cpp.o
examples/cpu-tutorials-matmul-sgemm-and-matmul-cpp: examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/build.make
examples/cpu-tutorials-matmul-sgemm-and-matmul-cpp: src/libdnnl.so.3.5
examples/cpu-tutorials-matmul-sgemm-and-matmul-cpp: /usr/lib/x86_64-linux-gnu/libm.so
examples/cpu-tutorials-matmul-sgemm-and-matmul-cpp: examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cpu-tutorials-matmul-sgemm-and-matmul-cpp"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/build: examples/cpu-tutorials-matmul-sgemm-and-matmul-cpp
.PHONY : examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/build

examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/clean:
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/examples && $(CMAKE_COMMAND) -P CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/cmake_clean.cmake
.PHONY : examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/clean

examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/depend:
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shreyas/G/shr-fuj/oneDNN_open_source /home/shreyas/G/shr-fuj/oneDNN_open_source/examples /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/examples /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/CMakeFiles/cpu-tutorials-matmul-sgemm-and-matmul-cpp.dir/depend

