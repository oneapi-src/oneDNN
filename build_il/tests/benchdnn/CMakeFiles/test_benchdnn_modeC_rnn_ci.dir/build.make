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

# Utility rule file for test_benchdnn_modeC_rnn_ci.

# Include any custom commands dependencies for this target.
include tests/benchdnn/CMakeFiles/test_benchdnn_modeC_rnn_ci.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/benchdnn/CMakeFiles/test_benchdnn_modeC_rnn_ci.dir/progress.make

tests/benchdnn/CMakeFiles/test_benchdnn_modeC_rnn_ci:

test_benchdnn_modeC_rnn_ci: tests/benchdnn/CMakeFiles/test_benchdnn_modeC_rnn_ci
test_benchdnn_modeC_rnn_ci: tests/benchdnn/CMakeFiles/test_benchdnn_modeC_rnn_ci.dir/build.make
.PHONY : test_benchdnn_modeC_rnn_ci

# Rule to build all files generated by this target.
tests/benchdnn/CMakeFiles/test_benchdnn_modeC_rnn_ci.dir/build: test_benchdnn_modeC_rnn_ci
.PHONY : tests/benchdnn/CMakeFiles/test_benchdnn_modeC_rnn_ci.dir/build

tests/benchdnn/CMakeFiles/test_benchdnn_modeC_rnn_ci.dir/clean:
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/benchdnn && $(CMAKE_COMMAND) -P CMakeFiles/test_benchdnn_modeC_rnn_ci.dir/cmake_clean.cmake
.PHONY : tests/benchdnn/CMakeFiles/test_benchdnn_modeC_rnn_ci.dir/clean

tests/benchdnn/CMakeFiles/test_benchdnn_modeC_rnn_ci.dir/depend:
	cd /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/shreyas/G/shr-fuj/oneDNN_open_source /home/shreyas/G/shr-fuj/oneDNN_open_source/tests/benchdnn /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/benchdnn /home/shreyas/G/shr-fuj/oneDNN_open_source/build_il/tests/benchdnn/CMakeFiles/test_benchdnn_modeC_rnn_ci.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/benchdnn/CMakeFiles/test_benchdnn_modeC_rnn_ci.dir/depend

