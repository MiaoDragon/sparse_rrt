cmake_minimum_required(VERSION 2.8)
project(SparseRRT)

list(APPEND CMAKE_PREFIX_PATH
    ${PROJECT_SOURCE_DIR}/deps/libtorch
    /usr/local/cuda-10.1)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_CXX_FLAGS "-std=c++1y ${CMAKE_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "-std=c++11 -D_GLIBCXX_USE_CXX11_ABI=1 ${CMAKE_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "-std=c++1y ${CMAKE_CXX_FLAGS}")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
SET(CMAKE_C_LINK_EXECUTABLE ${CMAKE_CXX_LINK_EXECUTABLE})


set(INCLUDE_DIRECTORIES include)

add_definitions(-O3 )

IF(APPLE)
   set(CMAKE_MACOSX_RPATH 1)
ENDIF(APPLE)

# Add dependency of trajopt
#list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/deps/trajopt/cmake/modules")
#find_package(Eigen3 REQUIRED)
#find_package(Boost COMPONENTS system python thread program_options REQUIRED)
#include_directories(${EIGEN3_INCLUDE_DIR})

#link_directories(${CMAKE_SOURCE_DIR}/deps/trajopt/build/lib)
#list(APPEND CMAKE_PREFIX_PATH  "${CMAKE_SOURCE_DIR}/deps/trajopt/build/lib")
#find_library(SCO_LIB sco)
#find_library(UTILS_LIB utils)



# Instead of find_package(PythonLibs REQUIRED), we manually search for python to make sure we link to the active version
# (as opposed to system python if brew python is installed)

#execute_process(
#    COMMAND "python2.7" -c "import sys;print(sys.prefix)"
#    OUTPUT_VARIABLE PYTHON_PREFIX
#    OUTPUT_STRIP_TRAILING_WHITESPACE
#)

# the below and the similar one further down depends on the python version
# arclabdl1: use python
# desktop: use python3.5
execute_process(
    COMMAND "python" -c "import sys;print(sys.prefix)"
    OUTPUT_VARIABLE PYTHON_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(APPLE)
    set(PYTHON_LIBRARIES "${PYTHON_PREFIX}/lib/libpython2.7.dylib")
else(APPLE)
    # it seems like libpython2.7.so is not necessary on ubuntu
    # set(PYTHON_LIBRARIES "${PYTHON_PREFIX}/lib/libpython2.7.so")

    #set(PYTHON_LIBRARIES "/usr/lib/x86_64-linux-gnu/libpython2.7.so")

    # the below depends where libpython is located on the system
    # arclabdl1:
    set(PYTHON_LIBRARIES "/home/arclabdl1/anaconda3/envs/ylmiao_kino/lib/libpython3.6m.so")
    # desktop:
    #set(PYTHON_LIBRARIES "/usr/lib/x86_64-linux-gnu/libpython3.5m.so")
endif(APPLE)


execute_process(
    COMMAND "python" -c "from distutils import sysconfig; print(sysconfig.get_python_inc())"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

#execute_process(
#    COMMAND "python2.7" -c "from distutils import sysconfig; print(sysconfig.get_python_inc())"
#    OUTPUT_VARIABLE PYTHON_INCLUDE_DIRS
#    OUTPUT_STRIP_TRAILING_WHITESPACE
#)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-10.1")
find_package(Torch REQUIRED)


include_directories(
        # ${Numpy_INCLUDE_DIRS}  # this should come before PYTHON includes because it also contains system numpy
        ${PYTHON_INCLUDE_DIRS}
        "${PROJECT_SOURCE_DIR}/include"
        "${PROJECT_SOURCE_DIR}/deps/pybind11/include"
        ${TORCH_INCLUDE_DIRS}
        )
message("found torch library path: " "${TORCH_LIBRARIES}")

set(INCLUDE_DIR
    ${PYTHON_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/deps/pybind11/include
    ${TORCH_INCLUDE_DIRS})

set(SOURCE_FILES
        src/motion_planners/rrt.cpp
        src/motion_planners/sst.cpp
        src/nearest_neighbors/graph_nearest_neighbors.cpp
        src/systems/car.cpp
        src/systems/cart_pole.cpp
        src/systems/cart_pole_obs.cpp
        src/systems/pendulum.cpp
        src/systems/point.cpp
        src/systems/rally_car.cpp
        src/systems/two_link_acrobot.cpp
        src/systems/two_link_acrobot_obs.cpp
        src/utilities/random.cpp
        src/utilities/timer.cpp
        src/image_creation/svg_image.cpp
        src/image_creation/planner_visualization.cpp
        src/systems/distance_functions.cpp
        src/bvp/psopt_bvp.cpp
        src/bvp/psopt_cart_pole.cpp
        src/bvp/psopt_pendulum.cpp
        src/bvp/psopt_acrobot.cpp
        src/bvp/psopt_point.cpp
        src/bvp/psopt_system.cpp
        src/neural/neural_deep_smp.cpp
)

add_library(${PROJECT_NAME} STATIC
    ${SOURCE_FILES})

add_library(sst_module SHARED
    ${PROJECT_SOURCE_DIR}/src/python_wrapper.cpp
    ${SOURCE_FILES})

# Add compilation for psopt
include(${CMAKE_SOURCE_DIR}/psopt_CMakeLists.txt)
message("D_GLIBCXX_USE_CXX11_ABI" "${D_GLIBCXX_USE_CXX11_ABI}")
message(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

#target_link_libraries(${PROJECT_NAME} SCO_LIB UTILS_LIB)
target_link_libraries(sst_module ${PYTHON_LIBRARIES} ${PROJECT_NAME} ${TORCH_LIBRARIES})


# Don't prepend wrapper library name with lib and add to Python libs.
set_target_properties(sst_module PROPERTIES
        PREFIX "_"
        SUFFIX ".so"
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/sparse_rrt
        )
