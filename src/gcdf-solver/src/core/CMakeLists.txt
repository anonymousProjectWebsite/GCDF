# add the solver as a library, the solver only has one cpp file and others are header files only.
add_library(CRISP
  src/cppad_core/CppAdInterface.cpp
)

target_include_directories(CRISP PUBLIC
  include                
  ${EIGEN3_INCLUDE_DIR}     
  ${Boost_INCLUDE_DIRS}
  ${CPPAD_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}      
)


target_link_libraries(CRISP
  ${Boost_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${CPPAD_LIBRARIES}
  yaml-cpp
  piqp::piqp # the piqp library (header file library)
  -ldl
  casadi::casadi # link to casadi
)



pybind11_add_module(pyCRISP python/pybindSolver.cpp)
# Linking the static library
target_link_libraries(pyCRISP PRIVATE CRISP)
# Ensure position-independent code for the shared module
set_target_properties(CRISP pyCRISP PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(pyCRISP PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${LIB_OUTPUT_DIR}
)


include(CMakePackageConfigHelpers)


write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/CRISPConfigVersion.cmake"
  VERSION 1.0.0
  COMPATIBILITY AnyNewerVersion
)


configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/CRISPConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/CRISPConfig.cmake"
  INSTALL_DESTINATION lib/cmake/CRISP
)


install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/CRISPConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/CRISPConfigVersion.cmake"
  DESTINATION lib/cmake/CRISP
)

install(DIRECTORY include/ DESTINATION include)


install(TARGETS CRISP
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)


