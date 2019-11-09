cmake_minimum_required(VERSION 2.8)
# Build a Python extension module using pybind11
#   pybindings_add_module(<module>)
# Here <module> should be the fully qualified name for the module,
# e.g. pybindings_add_module(foo.bar._baz)
# <module> becomes the target name in case you wish to do something to it later
# The source for the binding *must* be placed in src/pybindings/{relpath}/py{name}.cc
# E.g. for module=foo.bar._baz -> src/pybindings/bar/py_baz.cc

# adapted from https://gist.github.com/hovren/5b62175731433c741d07ee6f482e2936

set(module sparse_rrt._sst_module)
message("inside CMakeLists of pybind")
set(target_name ${module})
string(REPLACE "." "/" modpath ${module})
string(REPLACE "." ";" modlist ${module})

# The module name is the last entry
list(GET modlist -1 modname)

# Remove everything that is not the root or the module name
#list(REMOVE_AT modlist 0)
list(REMOVE_AT modlist -1)

# Get the relative path
if(modlist)
    string(REPLACE ";" "/" relpath "${modlist}")
else()
    set(relpath "")
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/deps/pybind11)
# Define the binding source file
set(sources src/python_wrapper.cpp)
#set(sources src/pybindings/${relpath}/py${modname}.cc)
# include definition of cpp module
include(cpp_CMakeLists.txt)
#include(psopt_CMakeLists.txt)
# Invoke pybind11 and set where the library should go, and what it is called
message("CMAKE_LIBRARY_OUTPUT_DIRECTORY: " ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
pybind11_add_module(${target_name} ${sources})
set(outdir ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${relpath})
target_link_libraries(${target_name} sst_module)  # link the cpp library to the pybinding
set_target_properties(${target_name} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${outdir})
set_target_properties(${target_name} PROPERTIES OUTPUT_NAME ${modname})
