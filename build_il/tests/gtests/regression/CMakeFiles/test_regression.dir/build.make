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
include tests/gtests/regression/CMakeFiles/test_regression.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/gtests/regression/CMakeFiles/test_regression.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/gtests/regression/CMakeFiles/test_regression.dir/progress.make

# Include the compile flags for this target's objects.
include tests/gtests/regression/CMakeFiles/test_regression.dir/flags.make

tests/gtests/regression/CMakeFiles/test_regression.dir/test_binary_stride.cpp.o: tests/gtests/regression/CMakeFiles/test_regression.dir/flags.make
tests/gtests/regression/CMakeFiles/test_regression.dir/test_binary_stride.cpp.o: ../tests/gtests/regression/test_binary_stride.cpp
tests/gtests/regression/CMakeFiles/test_regression.dir/test_binary_stride.cpp.o: tests/gtests/regression/CMakeFiles/test_regression.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/gtests/regression/CMakeFiles/test_regression.dir/test_binary_stride.cpp.o"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/regression && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/regression/CMakeFiles/test_regression.dir/test_binary_stride.cpp.o -MF CMakeFiles/test_regression.dir/test_binary_stride.cpp.o.d -o CMakeFiles/test_regression.dir/test_binary_stride.cpp.o -c /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/regression/test_binary_stride.cpp

tests/gtests/regression/CMakeFiles/test_regression.dir/test_binary_stride.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_regression.dir/test_binary_stride.cpp.i"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/regression && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/regression/test_binary_stride.cpp > CMakeFiles/test_regression.dir/test_binary_stride.cpp.i

tests/gtests/regression/CMakeFiles/test_regression.dir/test_binary_stride.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_regression.dir/test_binary_stride.cpp.s"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/regression && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/regression/test_binary_stride.cpp -o CMakeFiles/test_regression.dir/test_binary_stride.cpp.s

tests/gtests/regression/CMakeFiles/test_regression.dir/__/main.cpp.o: tests/gtests/regression/CMakeFiles/test_regression.dir/flags.make
tests/gtests/regression/CMakeFiles/test_regression.dir/__/main.cpp.o: ../tests/gtests/main.cpp
tests/gtests/regression/CMakeFiles/test_regression.dir/__/main.cpp.o: tests/gtests/regression/CMakeFiles/test_regression.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object tests/gtests/regression/CMakeFiles/test_regression.dir/__/main.cpp.o"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/regression && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/regression/CMakeFiles/test_regression.dir/__/main.cpp.o -MF CMakeFiles/test_regression.dir/__/main.cpp.o.d -o CMakeFiles/test_regression.dir/__/main.cpp.o -c /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/main.cpp

tests/gtests/regression/CMakeFiles/test_regression.dir/__/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_regression.dir/__/main.cpp.i"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/regression && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/main.cpp > CMakeFiles/test_regression.dir/__/main.cpp.i

tests/gtests/regression/CMakeFiles/test_regression.dir/__/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_regression.dir/__/main.cpp.s"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/regression && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/main.cpp -o CMakeFiles/test_regression.dir/__/main.cpp.s

tests/gtests/regression/CMakeFiles/test_regression.dir/__/__/test_thread.cpp.o: tests/gtests/regression/CMakeFiles/test_regression.dir/flags.make
tests/gtests/regression/CMakeFiles/test_regression.dir/__/__/test_thread.cpp.o: ../tests/test_thread.cpp
tests/gtests/regression/CMakeFiles/test_regression.dir/__/__/test_thread.cpp.o: tests/gtests/regression/CMakeFiles/test_regression.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object tests/gtests/regression/CMakeFiles/test_regression.dir/__/__/test_thread.cpp.o"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/regression && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/gtests/regression/CMakeFiles/test_regression.dir/__/__/test_thread.cpp.o -MF CMakeFiles/test_regression.dir/__/__/test_thread.cpp.o.d -o CMakeFiles/test_regression.dir/__/__/test_thread.cpp.o -c /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/test_thread.cpp

tests/gtests/regression/CMakeFiles/test_regression.dir/__/__/test_thread.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_regression.dir/__/__/test_thread.cpp.i"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/regression && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/test_thread.cpp > CMakeFiles/test_regression.dir/__/__/test_thread.cpp.i

tests/gtests/regression/CMakeFiles/test_regression.dir/__/__/test_thread.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_regression.dir/__/__/test_thread.cpp.s"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/regression && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/test_thread.cpp -o CMakeFiles/test_regression.dir/__/__/test_thread.cpp.s

# Object files for target test_regression
test_regression_OBJECTS = \
"CMakeFiles/test_regression.dir/test_binary_stride.cpp.o" \
"CMakeFiles/test_regression.dir/__/main.cpp.o" \
"CMakeFiles/test_regression.dir/__/__/test_thread.cpp.o"

# External object files for target test_regression
test_regression_EXTERNAL_OBJECTS =

tests/gtests/regression/test_regression: tests/gtests/regression/CMakeFiles/test_regression.dir/test_binary_stride.cpp.o
tests/gtests/regression/test_regression: tests/gtests/regression/CMakeFiles/test_regression.dir/__/main.cpp.o
tests/gtests/regression/test_regression: tests/gtests/regression/CMakeFiles/test_regression.dir/__/__/test_thread.cpp.o
tests/gtests/regression/test_regression: tests/gtests/regression/CMakeFiles/test_regression.dir/build.make
tests/gtests/regression/test_regression: src/libdnnl.so.3.5
tests/gtests/regression/test_regression: tests/gtests/gtest/libdnnl_gtest.a
tests/gtests/regression/test_regression: tests/gtests/regression/CMakeFiles/test_regression.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable test_regression"
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/regression && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_regression.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/gtests/regression/CMakeFiles/test_regression.dir/build: tests/gtests/regression/test_regression
.PHONY : tests/gtests/regression/CMakeFiles/test_regression.dir/build

tests/gtests/regression/CMakeFiles/test_regression.dir/clean:
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/regression && $(CMAKE_COMMAND) -P CMakeFiles/test_regression.dir/cmake_clean.cmake
.PHONY : tests/gtests/regression/CMakeFiles/test_regression.dir/clean

tests/gtests/regression/CMakeFiles/test_regression.dir/depend:
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shreyas/G/shr-fuj/oneDNN_open_source /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/gtests/regression /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/regression /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/gtests/regression/CMakeFiles/test_regression.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/gtests/regression/CMakeFiles/test_regression.dir/depend

