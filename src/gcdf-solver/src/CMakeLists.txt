cmake_minimum_required(VERSION 3.10)
project(SolverDevelopment)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# Add PKG_CONFIG_PATH
set(ENV{PKG_CONFIG_PATH} "/usr/local/share/pkgconfig:$ENV{PKG_CONFIG_PATH}")
set(GEN_OUTPUT_DIR "/home/robot/workspace/src/gcdf-solver/scripts/_l4c_generated")
set(GEN_CORE_DIR    "/home/robot/workspace/src/gcdf-solver/core")   # cdf_func.c and cdf_func.h
set(LIB_OUTPUT_DIR "/home/robot/workspace/src/gcdf-solver/lib")  
    
# Find packages
find_package(PkgConfig REQUIRED)
# CPPAD output the .pc file, find it with pkgconfig
# CPPADCG is header file only, if you install it default, you don't need to find it explicitly
pkg_check_modules(CPPAD REQUIRED cppad)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS
system
filesystem
log_setup
log)
find_package(yaml-cpp REQUIRED) # Yaml for reading the hyper-parameters for the solver
find_package(piqp REQUIRED) # piqp is header only
message(STATUS "PIQP include dir: ${piqp_INCLUDE_DIRS}")
# set(CMAKE_PREFIX_PATH "/usr/local" ${CMAKE_PREFIX_PATH})
find_package(casadi CONFIG REQUIRED)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# print if find eigen and boost, and pkgconfig
message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Boost include dir: ${Boost_INCLUDE_DIRS}")
message(STATUS "CPPAD include dir: ${CPPAD_INCLUDE_DIRS}")
message(STATUS "YAML-CPP include dir: ${YAML_CPP_INCLUDE_DIR}")


add_library(cdf_func SHARED "${GEN_CORE_DIR}/gcdf_func.c")
# set library output directory
set_target_properties(cdf_func PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${LIB_OUTPUT_DIR}
)
target_include_directories(cdf_func PUBLIC ${GEN_CORE_DIR})
target_link_directories(cdf_func PUBLIC
${GEN_OUTPUT_DIR}
)
target_link_libraries(cdf_func cdf)
add_subdirectory(core)

